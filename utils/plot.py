from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_score_histograms(model_name: str, id_train_scores: np.ndarray, id_test_scores: np.ndarray, ood_test_scores: np.ndarray) -> None:
    def plot_hist(set_name, score_vector):
        plt.title(set_name)
        cutoff = np.percentile(score_vector, 95)
        plt.xlim(0, cutoff)
        plt.hist(score_vector, density=True)
        _save_figure(model_name.lower().replace(' ', '_') + '_' + set_name.lower().replace(' ', '_') +'_scores.jpg')
        plt.clf()
    plot_hist('ID Training', id_train_scores)
    plot_hist('ID Test', id_test_scores)
    plot_hist('OOD Test', ood_test_scores)

def plot_roc_curve(model_name: str, y_true: np.ndarray, y_scores: np.ndarray):
    plt.title(model_name +" ROC Curve")
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
    auc = round(metrics.auc(fpr, tpr), 3)
    plt.plot(fpr, tpr, label=f'AUC={auc}')
    plt.legend()
    _save_figure(model_name.lower().replace(' ', '_') +'_roc.jpg')
    plt.clf()

def plot_pr_curve(model_name: str, y_true: np.ndarray, y_scores: np.ndarray):
    plt.title(model_name +" Precision-Recall Curve")
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores)
    ap = round(metrics.average_precision_score(y_true, y_scores), 3)
    plt.plot(recall, precision, label=f'AP={ap}')
    plt.legend()
    _save_figure(model_name.lower().replace(' ', '_') +'_pr.jpg')
    plt.clf()

def _save_figure(filename: str):
    if not (os.path.exists('plots') and os.path.isdir('plots')):
        os.mkdir('plots')
    plt.savefig(os.path.join('plots', filename))
