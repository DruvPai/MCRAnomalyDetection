from anomaly_detection import AnomalyDetectionAsymptoticMICL
from torch.utils.data import DataLoader
import torch
from sklearn import metrics
import train
import utils.corruption
import utils.dataset
import utils.model
import utils.plot
import numpy as np

def ood_labels(in_scores, out_scores):
    y_scores = np.concatenate((in_scores, out_scores), axis=0)
    y_true = np.zeros(len(y_scores))
    y_true[len(in_scores):] = 1
    return y_scores, y_true

def correct(in_scores, out_scores):
    y_scores, y_true = ood_labels(in_scores, out_scores)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
    index = np.searchsorted(tpr, 0.95)
    delta = thresholds[index]

    return np.asarray(out_scores) < delta

def ood_metrics(in_scores, out_scores):
    avg_in = np.mean(in_scores)
    avg_out = np.mean(out_scores)

    y_scores, y_true = ood_labels(in_scores, out_scores)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
    AUROC = metrics.auc(fpr, tpr)

    index = np.searchsorted(tpr, 0.95)
    TNR = 1 - fpr[index]
    delta = thresholds[index]

    DTACC = np.max(tpr * len(in_scores) + (1 - fpr) * len(out_scores)) / len(y_scores)

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores)
    AUIN = metrics.auc(recall, precision)

    # precision-recall curve with out distribution positive
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, -y_scores, pos_label=0)
    AUOUT = metrics.auc(recall, precision)

    return {'avg_in': avg_in, 'avg_out': avg_out, 'AUROC': AUROC, 'AUIN': AUIN,
            'AUOUT': AUOUT, 'TNR': TNR, 'DTACC': DTACC}


def anomaly_detection(model_dir: str, out: str, epoch: int = None, data_dir: str = './data/'):
    params = utils.model.load_params(model_dir)
    net, epoch = utils.model.load_checkpoint(model_dir, epoch, eval_=True)
    net = net.cuda().eval()

    # get train data
    train_transforms = utils.dataset.load_transforms('test')
    trainset = utils.dataset.load_trainset(params['data'], train_transforms, train=True, path=data_dir)
    trainloader = DataLoader(trainset, batch_size=500)
    train_features, train_labels = utils.model.get_features(net, trainloader)

    anomaly_detector = AnomalyDetectionAsymptoticMICL()
    train_scores = anomaly_detector.fit_predict(train_features, train_labels, params['eps'])

    # get test features and labels
    test_transforms = utils.dataset.load_transforms('test')
    testset = utils.dataset.load_trainset(params['data'], test_transforms, train=False)
    testloader = DataLoader(testset, batch_size=500)
    test_features, test_labels = utils.model.get_features(net, testloader)

    # get out features and labels
    out_transforms = utils.dataset.load_transforms('test')
    outset = utils.dataset.load_trainset(out, out_transforms, train=False)
    outloader = DataLoader(outset, batch_size=500)
    out_features, out_labels = utils.model.get_features(net, outloader)

    train_scores = train_scores.cpu().numpy()

    in_scores = anomaly_detector.predict(test_features)
    in_scores = in_scores.cpu().numpy()

    out_scores = anomaly_detector.predict(out_features)
    out_scores = out_scores.cpu().numpy()

    return train_scores, in_scores, out_scores


# train.supervised_train({'arch': 'resnet18', 'data': 'cifar10', 'fd': 128, 'epo': 20, 'bs': 500, 'eps': 0.5, 'gam1': 1., 'gam2': 1, 'lr': 0.01, 'transform': 'cifar'})
model_dir = './saved_models/sup_resnet18+128_cifar10_epo20_bs500_lr0.01_mom0.9_wd0.0005_gam11.0_gam21.0_eps0.5_lcr0.0'
# model_dir = './saved_models/sup_resnet18+128_cifar10_epo500_bs1000_lr0.01_mom0.9_wd0.0005_gam11.0_gam21.0_eps0.5_lcr0.0'
out = "svhn"
train_scores, in_scores, out_scores = anomaly_detection(model_dir, out)
print(ood_metrics(in_scores, out_scores))

