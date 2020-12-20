from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

def plot_score_histograms(model_name: str, id_train_scores: np.ndarray, id_test_scores: np.ndarray, ood_test_scores: np.ndarray) -> None:
    plt.title(model_name + " scores")
    plt.hist(id_train_scores, density=True, bins="sqrt", label='ID Train', lw=3, alpha=0.5, color='k')
    plt.hist(id_test_scores, density=True, bins="sqrt", label='ID Test', lw=3, alpha=0.5, color='b')
    plt.hist(ood_test_scores, density=True, bins="sqrt", label='OOD Test', lw=3, alpha=0.5, color='r')
    plt.legend()
    plt.savefig(model_name.lower().replace(' ', '_') + '_' + 'scores.jpg')
    plt.clf()

def plot_roc_curve(model_name: str, y_true: np.ndarray, y_scores: np.ndarray):
    plt.title(model_name +" ROC Curve")
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr, 'r')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.savefig(model_name.lower().replace(' ', '_') +'_roc.jpg')
    plt.clf()

def plot_pr_curve(model_name: str, y_true: np.ndarray, y_scores: np.ndarray):
    plt.title(model_name +" Precision-Recall Curve")
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores)
    plt.plot(recall, precision)
    plt.savefig(model_name.lower().replace(' ', '_') +'_pr.jpg')
    plt.clf()
