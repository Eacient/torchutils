import numpy as np
import torch

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        # mask = (label_true >= 0) & (label_true < self.num_classes)
        mask = (label_true >= 0) & (label_true < self.num_classes) & (label_pred < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        recall = np.diag(self.hist) / self.hist.sum(axis=1)
        # recall = np.nanmean(recall)
        precision = np.diag(self.hist) / self.hist.sum(axis=0)
        # precision = np.nanmean(precision)
        TP = np.diag(self.hist)
        TN = self.hist.sum(axis=1) - np.diag(self.hist)
        FP = self.hist.sum(axis=0) - np.diag(self.hist)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_classes), iu))

        return acc, recall, precision, TP, TN, FP, cls_iu, mean_iu, fwavacc
    
def miou(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes) & (label_pred < num_classes)
    hist = torch.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    acc = torch.diag(hist).sum() / hist.sum()
    recall = torch.diag(hist) / hist.sum(axis=1)
    recall = torch.nanmean(recall)
    precision = torch.diag(hist) / hist.sum(axis=0)
    precision = torch.nanmean(precision)
    # TP = np.diag(hist)
    # TN = hist.sum(axis=1) - np.diag(hist)
    # FP = hist.sum(axis=0) - np.diag(hist)
    iu = torch.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - torch.diag(hist))
    mean_iu = torch.nanmean(iu)
    freq = torch.sum(axis=1) / torch.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    # cls_iu = dict(zip(range(num_classes), iu))
    return acc.item(), recall.item(), precision.item(), mean_iu.item(), fwavacc.item()