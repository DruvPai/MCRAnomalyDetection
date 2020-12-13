from anomaly_detection import AnomalyDetection
from torch.utils.data import DataLoader
import train
import utils.corruption
import utils.dataset
import utils.model

def anomaly_detection(model_dir: str, epoch: int=None, data_dir: str='./data/', alpha: float=0.05):
    params = utils.model.load_params(model_dir)
    net, epoch = utils.model.load_checkpoint(model_dir, epoch, eval_=True)
    net = net.cuda().eval()

    # get train data
    train_transforms = utils.dataset.load_transforms('test')
    trainset = utils.dataset.load_trainset(params['data'], train_transforms, train=True, path=data_dir)
    if 'label_corruption_ratio' in params.keys(): # supervised corruption case
        trainset = utils.corruption.corrupt_labels(params['corruption'])(trainset, params['label_corruption_ratio'], params['label_corruption_seed'])
    new_labels = trainset.targets
    trainloader = DataLoader(trainset, batch_size=200)
    train_features, train_labels = utils.model.get_features(net, trainloader)

    # get test features and labels
    test_transforms = utils.dataset.load_transforms('test')
    testset = utils.dataset.load_trainset(params['data'], test_transforms, train=False)
    testloader = DataLoader(testset, batch_size=200)
    test_features, test_labels = utils.model.get_features(net, testloader)

    anomaly_detector = AnomalyDetection(alpha)
    anomaly_detector.fit(train_features, train_labels, params['eps_sq'])
    anomaly_predictions = anomaly_detector.predict(test_features)
    anomaly_predictions = anomaly_predictions.cpu().numpy()
    test_labels = test_labels.cpu().numpy()
    return utils.model.compute_accuracy(anomaly_predictions, test_labels)


# train.supervised_train({'arch': 'resnet18', 'data': 'cifar10', 'feature_dim': 128, 'epochs': 20, 'batch_size': 500, 'eps_sq': 0.5, 'gamma_1': 1., 'gamma_2': 1., 'learning_rate': 0.01, 'label_corruption_ratio': 0.1})
print(anomaly_detection('./saved_models/sup_resnet18+128_cifar10_epo20_bs500_lr0.01_mom0.9_wd0.0005_gam11.0_gam21.0_eps0.5_lcr0.1'))
