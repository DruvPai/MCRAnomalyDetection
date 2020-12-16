import os
import json
import numpy as np
from tqdm import tqdm
import typing
import torch
import torchvision

def load_architectures(name, dim):
    """Returns a network architecture.

    Parameters:
        name (str): name of the architecture
        dim (int): feature dimension of vector presentation

    Returns:
        net (torch.nn.Module)

    """
    _name = name.lower()
    if _name == "resnet18":
        from architectures.resnet_cifar import ResNet18
        net = ResNet18(dim)
    elif _name == "resnet18ctrl":
        from architectures.resnet_cifar import ResNet18Control
        net = ResNet18Control(dim)
    elif _name == "resnet18stl":
        from architectures.resnet_stl import ResNet18STL
        net = ResNet18STL(dim)
    elif _name == "vgg11":
        from architectures.vgg_cifar import VGG11
        net = VGG11(dim)
    elif _name == "resnext29_2x64d":
        from architectures.resnext_cifar import ResNeXt29_2x64d
        net = ResNeXt29_2x64d(dim)
    elif _name == "resnext29_4x64d":
        from architectures.resnext_cifar import ResNeXt29_4x64d
        net = ResNeXt29_4x64d(dim)
    elif _name == "resnext29_8x64d":
        from architectures.resnext_cifar import ResNeXt29_8x64d
        net = ResNeXt29_8x64d(dim)
    elif _name == "resnext29_32x4d":
        from architectures.resnext_cifar import ResNeXt29_32x4d
        net = ResNeXt29_32x4d(dim)
    elif _name == "resnet10mnist":
        from architectures.resnet_mnist import ResNet10MNIST
        net = ResNet10MNIST(dim)
    else:
        raise NameError("{} not found in architectures.".format(name))
    net = torch.nn.DataParallel(net).cuda()
    return net


def load_checkpoint(model_dir, epoch=None, eval_=False):
    """Load checkpoint from model directory. Checkpoints should be stored in
    `model_dir/checkpoints/model-epochX.ckpt`, where `X` is the epoch number.

    Parameters:
        model_dir (str): path to model directory
        epoch (int): epoch number; set to None for last available epoch
        eval_ (bool): PyTorch evaluation mode. set to True for testing

    Returns:
        net (torch.nn.Module): PyTorch checkpoint at `epoch`
        epoch (int): epoch number

    """
    if epoch is None: # get last epoch
        ckpt_dir = os.path.join(model_dir, 'checkpoints')
        print(ckpt_dir)
        epochs = [int(e[11:-3]) for e in os.listdir(ckpt_dir) if e[-3:] == ".pt"]
        print(epochs)
        epoch = np.sort(epochs)[-1]
    ckpt_path = os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(epoch))
    params = load_params(model_dir)
    print('Loading checkpoint: {}'.format(ckpt_path))
    state_dict = torch.load(ckpt_path)
    net = load_architectures(params['arch'], params['fd'])
    net.load_state_dict(state_dict)
    del state_dict
    if eval_:
        net.eval()
    return net, epoch


def init_pipeline(model_dir, headers=None):
    """Initialize folder and .csv logger."""
    # project folder
    os.makedirs(model_dir)
    os.makedirs(os.path.join(model_dir, 'checkpoints'))
    os.makedirs(os.path.join(model_dir, 'figures'))
    os.makedirs(os.path.join(model_dir, 'plabels'))
    if headers is None:
        headers = ["epoch", "step", "loss", "discrimn_loss_e", "compress_loss_e",
                   "discrimn_loss_t",  "compress_loss_t"]
    create_csv(model_dir, 'losses.csv', headers)
    print("project dir: {}".format(model_dir))

def create_csv(model_dir, filename, headers):
    """Create .csv file with filename in model_dir, with headers as the first line
    of the csv. """
    csv_path = os.path.join(model_dir, filename)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w+') as f:
        f.write(','.join(map(str, headers)))
    return csv_path

def save_params(model_dir, params):
    """Save params to a .json file. Params is a dictionary of parameters."""
    path = os.path.join(model_dir, 'params.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)

def update_params(model_dir, pretrain_dir):
    """Updates architecture and feature dimension from pretrain directory
    to new directory. """
    params = load_params(model_dir)
    old_params = load_params(pretrain_dir)
    params['arch'] = old_params['arch']
    params['fd'] = old_params['fd']
    save_params(model_dir, params)

def load_params(model_dir):
    """Load params.json file in model directory and return dictionary."""
    _path = os.path.join(model_dir, "params.json")
    with open(_path, 'r') as f:
        _dict = json.load(f)
    return _dict

def save_state(model_dir, *entries, filename='losses.csv'):
    """Save entries to csv. Entries is list of numbers. """
    csv_path = os.path.join(model_dir, filename)
    assert os.path.exists(csv_path), 'CSV file is missing in project directory.'
    with open(csv_path, 'a') as f:
        f.write('\n'+','.join(map(str, entries)))

def save_ckpt(model_dir, net, epoch):
    """Save PyTorch checkpoint to ./checkpoints/ directory in model directory. """
    torch.save(net.state_dict(), os.path.join(model_dir, 'checkpoints',
                                              'model-epoch{}.pt'.format(epoch)))

def save_labels(model_dir, labels, epoch):
    """Save labels of a certain epoch to directory. """
    path = os.path.join(model_dir, 'plabels', f'epoch{epoch}.npy')
    np.save(path, labels)


def get_features(net, trainloader, verbose=True):
    '''Extract all features out into one single batch.

    Parameters:
        net (torch.nn.Module): get features using this model
        trainloader (torchvision.dataloader): dataloader for loading data
        verbose (bool): shows loading staus bar

    Returns:
        features (torch.tensor): with dimension (num_samples, feature_dimension)
        labels (torch.tensor): with dimension (num_samples, )
    '''
    features = []
    labels = []
    if verbose:
        train_bar = tqdm(trainloader, desc="extracting all features from dataset")
    else:
        train_bar = trainloader
    for step, (batch_imgs, batch_lbls) in enumerate(train_bar):
        batch_features = net(batch_imgs.cuda())
        features.append(batch_features.cpu().detach())
        labels.append(batch_lbls)
    return torch.cat(features).cuda(), torch.cat(labels).cuda()


def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size
