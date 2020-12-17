from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

def plot_score_histograms(model_name: str, id_train_scores: np.ndarray, id_test_scores: np.ndarray, ood_test_scores: np.ndarray) -> None:
    def plot_hist(set_name, score_vector):
        plt.title(set_name)
        plt.hist(score_vector, density=True)
        plt.savefig(model_name.lower().replace(' ', '_') + '_' + set_name.lower().replace(' ', '_') +'scores.jpg')
        plt.clf()
    plot_hist('ID Training', id_train_scores)
    plot_hist('ID Test', id_test_scores)
    plot_hist('OOD Test', ood_test_scores)

def plot_roc_curve(model_name: str, y_true: np.ndarray, y_scores: np.ndarray):
    plt.title(model_name +" ROC Curve")
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr)
    plt.savefig(model_name.lower().replace(' ', '_') +'_roc.jpg')
    plt.clf()

def plot_pr_curve(model_name: str, y_true: np.ndarray, y_scores: np.ndarray):
    plt.title(model_name +" Precision-Recall Curve")
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores)
    plt.plot(recall, precision)
    plt.savefig(model_name.lower().replace(' ', '_') +'_pr.jpg')
    plt.clf()

