# Source mapping and reproducibility notes

| Notebook | Python modules/scripts |
| --- | --- |
| `Copy_of_data_preparation.ipynb` | `dufunet/preparation.py`, `scripts/prepare_data.py`, `scripts/validate_tiles.py` |
| `cram-dualbranch-256-multiseed.ipynb` | `dufunet/{constants,dataset,model,losses,metrics,training}.py`, `scripts/{train,aggregate}.py` |
| `qualitative_analysis.ipynb` | `dufunet/visualization.py`, `scripts/qualitative.py` |

- Model, losses, dataset normalization and confusion metrics came from the **training notebook**. It sets Planet RGB → SegFormer-B2 and Sentinel-2 46-band stack → EfficientNet-B0. The training notebook's `ConcatAlign` **does** return its projected Sentinel feature `+ F_p`, as does CRAM. The qualitative notebook has a different duplicated model implementation; the new analysis script uses the shared training model and strict state-dict loading.
- The original preparation notebook made a random stratified tile split with 15% validation. The new preparation CLI makes contiguous vertical train/validation regions with 25% of tile columns for validation by default. This is a new dataset split, so resulting metrics must not be presented as a reproduction of the earlier notebook run. A separate spatial test region is still needed for independent final testing.
- The original preparation notebook calculated raw Planet/Sentinel statistics, but training normalized Planet after a global 2–98% stretch using hardcoded training constants. That computation is retained in the dataset module. The generated `images_stats.json` is informative, not substituted into training. The optional `--sentinel-stats` deliberately changes the experiment and must match during analysis.
- The 46-band order and 1–6 mask labels must match the source data. Mask 0 is mapped to ignore index 255 by the dataset loader. Synthetic zero fill for missing Sentinel values follows the original preparation notebook.
- `scripts/qualitative.py` plots fusion models only; the original analysis notebook's 5-model figure needs baseline checkpoints and their exact preprocessing. No baseline model weights were supplied. The original qualitative checkpoint paths ended in `.zip`; extract the actual `.pth` checkpoint before invoking the new script.
- The training notebook's CRAM uses local 5×5 attention, Q/K LayerNorm and a Planet residual. These classes were copied as Python modules. Concat and CRAM should be interpreted as full fusion variants, not a controlled test of attention alone.
- Record data acquisition, region polygons, masking, tile manifests, software versions, seeds, normalization, training settings, model selection, and final held-out test metrics before using this repository for paper reproducibility claims.
