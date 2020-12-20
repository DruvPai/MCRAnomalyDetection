from anomaly_detection import AnomalyDetection
from torch.utils.data import DataLoader
import torch
from sklearn import metrics
import train
import utils
import numpy as np
import pickle
from evaluate import ood_metrics, ood_labels

distr = ["cifar10", "cifar100", "svhn"]

in_dist = "cifar10"
out_dist = "cifar100"

filename = "scores/" + in_dist + "_vs_" + out_dist + ".pkl"
scores = pickle.load(open(filename, 'rb'))
train_scores = scores["train_scores"]
in_scores = scores["in_scores"]
out_scores = scores["out_scores"]
test_acc = scores["test_acc"]

stats = ood_metrics(in_scores, out_scores)
AUROC = stats["AUROC"]
print(in_dist + " vs " + out_dist + ": AUROC = " + str(AUROC) + " test_acc =" + str(test_acc))


y_scores, y_true = ood_labels(in_scores, out_scores)
utils.plot.plot_score_histograms('CIFAR10 vs CIFAR100', train_scores, in_scores, out_scores)
utils.plot.plot_roc_curve('CIFAR10 vs CIFAR100', y_true, y_scores)
# utils.plot.plot_pr_curve('CIFAR10 vs CIFAR100', y_true, y_scores)
print(ood_metrics(in_scores, out_scores))