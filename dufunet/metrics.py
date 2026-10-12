"""Six-class confusion matrix metrics."""
import numpy as np

class SegMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.C = num_classes; self.ignore_index = ignore_index; self.reset()
    def reset(self):
        self.conf_mat = np.zeros((self.C, self.C), dtype=np.int64)
    def update(self, preds, targets):
        mask = targets != self.ignore_index
        np.add.at(self.conf_mat, (targets[mask].ravel(), preds[mask].ravel()), 1)
    def compute(self):
        cm = self.conf_mat.astype(np.float64)
        tp = np.diag(cm); fp = cm.sum(0)-tp; fn = cm.sum(1)-tp
        iou = tp/(tp+fp+fn+1e-6); f1 = 2*tp/(2*tp+fp+fn+1e-6); oa = tp.sum()/(cm.sum()+1e-6)
        return {'mIoU':float(iou.mean()), 'IoU':iou.tolist(),
                'mF1':float(f1.mean()),   'F1':f1.tolist(), 'OA':float(oa)}

