"""Fusion architecture ported from training notebook cell 18."""
import torch
from torch import nn
from torch.nn import functional as F
import timm
from transformers import SegformerModel, SegformerConfig

def build_local_mask(H, W, window=5, device='cpu'):
    N = H * W
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    coords = torch.stack([ys.flatten(), xs.flatten()], dim=1).to(device)   # [N,2]
    dist = (coords[:,None,:] - coords[None,:,:]).abs().max(-1).values      # Chebyshev distance [N,N]
    mask = torch.where(dist <= window // 2, 0.0, float('-inf'))
    return mask


class CRAM(nn.Module):
    def __init__(self, planet_dim, sentinel_dim, proj_dim=256, num_heads=8, window=5):
        super().__init__()
        self.window = window
        self.q_proj = nn.Conv2d(planet_dim,   proj_dim, 1)
        self.k_proj = nn.Conv2d(sentinel_dim, proj_dim, 1)
        self.v_proj = nn.Conv2d(sentinel_dim, proj_dim, 1)
        self.q_norm = nn.LayerNorm(proj_dim)   
        self.k_norm = nn.LayerNorm(proj_dim)
        self.attn = nn.MultiheadAttention(proj_dim, num_heads, batch_first=True, dropout=0.1)
        self.out_proj = nn.Sequential(nn.Conv2d(proj_dim, planet_dim, 1), nn.BatchNorm2d(planet_dim))
        self._mask_cache = {}

    def forward(self, F_p, F_s):
        B, Cp, Hp, Wp = F_p.shape
        F_s_up = F.interpolate(F_s, size=(Hp,Wp), mode='bilinear', align_corners=False)

        Q = self.q_norm(self.q_proj(F_p).flatten(2).transpose(1,2))
        K = self.k_norm(self.k_proj(F_s_up).flatten(2).transpose(1,2))
        V = self.v_proj(F_s_up).flatten(2).transpose(1,2)

        key = (Hp, Wp, self.window, F_p.device)
        if key not in self._mask_cache:
            self._mask_cache[key] = build_local_mask(Hp, Wp, self.window, F_p.device)
        mask = self._mask_cache[key]

        attn_out, _ = self.attn(Q, K, V, attn_mask=mask)
        attn_out = attn_out.transpose(1,2).reshape(B, -1, Hp, Wp)
        return self.out_proj(attn_out) + F_p


class ConcatAlign(nn.Module):
    """
    Fair concat baseline: upsample Sentinel feature and project it to Planet dim.
    No cross-attention. Same FusionBlock/decoder as CRAM model.
    """
    def __init__(self, planet_dim, sentinel_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(sentinel_dim, planet_dim, 1, bias=False),
            nn.BatchNorm2d(planet_dim),
            nn.GELU(),
        )

    # def forward(self, F_p, F_s):
    #     Hp, Wp = F_p.shape[-2:]
    #     F_s_up = F.interpolate(F_s, size=(Hp, Wp), mode='bilinear', align_corners=False)
    #     return self.proj(F_s_up)                              # [B,512,Hp,Wp]
    def forward(self, F_p, F_s):
        Hp, Wp = F_p.shape[-2:]
        F_s_up = F.interpolate(F_s, size=(Hp, Wp), mode='bilinear', align_corners=False)
        return self.proj(F_s_up) + F_p 


class FusionBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.residual = nn.Conv2d(in_dim, out_dim, 1, bias=False) if in_dim != out_dim else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) + self.residual(x))


class SegFormerMLPDecoder(nn.Module):
    def __init__(self, in_channels=[64,128,320,512], embed_dim=256, num_classes=6):
        super().__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(c, embed_dim) for c in in_channels])
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features):
        B, _, H, W = features[0].shape
        outs = []
        for feat, lin in zip(features, self.linear_layers):
            x = feat.permute(0,2,3,1)
            x = lin(x)
            x = x.permute(0,3,1,2)
            if x.shape[-2:] != (H, W):
                x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False)
            outs.append(x)
        x = torch.cat(outs, dim=1)
        x = self.linear_fuse(x)
        x = self.dropout(x)
        return self.linear_pred(x)


class DualBranchFusion(nn.Module):
    SF_B2_CHANNELS = [64, 128, 320, 512]
    EFF_B0_CHANNELS = 320

    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.fusion_mode = cfg['fusion_mode']

        self.planet_encoder = (SegformerModel.from_pretrained(
            cfg['planet_backbone'], output_hidden_states=True,
            ignore_mismatched_sizes=True,
        ) if pretrained else SegformerModel(SegformerConfig(
            depths=[3, 4, 6, 3], hidden_sizes=self.SF_B2_CHANNELS,
            num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1],
            output_hidden_states=True,
        )))

        self.sentinel_encoder = timm.create_model(
            cfg['sentinel_backbone'],
            pretrained=pretrained,
            features_only=True,
            in_chans=cfg['sentinel_channels'],
            out_indices=(4,),
        )

        if cfg['fusion_mode'] == 'cram':
            self.aligner = CRAM(
                planet_dim=self.SF_B2_CHANNELS[-1],
                sentinel_dim=self.EFF_B0_CHANNELS,
                proj_dim=cfg['cram_dim'],
                num_heads=cfg['cram_heads'],
            )
        elif cfg['fusion_mode'] == 'concat':
            self.aligner = ConcatAlign(
                planet_dim=self.SF_B2_CHANNELS[-1],
                sentinel_dim=self.EFF_B0_CHANNELS,
            )
        else:
            raise ValueError(f'Unknown fusion_mode: {cfg["fusion_mode"]}')

        self.fusion = FusionBlock(
            in_dim=self.SF_B2_CHANNELS[-1] * 2,
            out_dim=self.SF_B2_CHANNELS[-1],
        )
        self.decoder = SegFormerMLPDecoder(
            in_channels=self.SF_B2_CHANNELS,
            embed_dim=256,
            num_classes=cfg['num_classes'],
        )

    def forward(self, planet, sentinel):
        planet_out = self.planet_encoder(pixel_values=planet, output_hidden_states=True)
        planet_feats = list(planet_out.hidden_states)
        F_p = planet_feats[-1]                              # [B,512,Hp/32,Wp/32]

        sentinel_feats = self.sentinel_encoder(sentinel)
        F_s = sentinel_feats[-1]                            # [B,320,Hs/32,Ws/32]

        F_s_aligned = self.aligner(F_p, F_s)                # CRAM or concat aligner
        F_fused = self.fusion(torch.cat([F_p, F_s_aligned], dim=1))
        decoder_feats = planet_feats[:3] + [F_fused]
        return self.decoder(decoder_feats)

