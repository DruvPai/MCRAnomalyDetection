import inspect
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.augmentation
import utils.corruption
import utils.dataset
import utils.model
from mcr import MaximalCodingRateReduction


def supervised_train(params: dict) -> None:
    # (arch: str = 'resnet18', feature_dim: int = 128, data: str = 'cifar10', epochs: int = 800,
    #      batch_size: int = 1000, learning_rate: float = 0.001, momentum: float = 0.9,
    #      weight_decay: float = 5e-4, gamma_1: float = 1.,
    #      gamma_2: float = 1., eps_sq: float = 0.5, corruption: str = 'default',
    #      label_corruption_ratio: float = 0.,
    #      label_corruption_seed: int = 10, tail: str = '', transform: str = 'default',
    #      save_dir: str = './saved_models/',
    #      data_dir: str = './data/', pretrain_dir: str = None, pretrain_epoch: int = None)
    arch = params.get('arch', 'resnet18')
    feature_dim = params.get('fd', 128)
    data = params.get('data', 'cifar10')
    epochs = params.get('epo', 800)
    batch_size = params.get('bs', 1000)
    learning_rate = params.get('lr', 0.001)
    momentum = params.get('mom', 0.9)
    weight_decay = params.get('wd', 5e-4)
    gamma_1 = params.get('gamma_1', 1.)
    gamma_2 = params.get('gamma_2', 1.)
    eps_sq = params.get('eps', 0.5)
    corruption = params.get('corrupt', 'default')
    label_corruption_ratio = params.get('lcr', 0.)
    label_corruption_seed = params.get('lcs', 10)
    tail = params.get('tail', '')
    transform = params.get('transform', 'default')
    save_dir = params.get('save_dir', './saved_models/')
    data_dir = params.get('data_dir', './data/')
    pretrain_dir = params.get('pretrain_dir', None)
    pretrain_epoch = params.get('pretrain_epo', None)
    params['arch'] = arch
    params['fd'] = feature_dim
    params['data'] = data
    params['epo'] = epochs
    params['bs'] = batch_size
    params['lr'] = learning_rate
    params['mom'] = momentum
    params['wd'] = weight_decay
    params['gam1'] = gamma_1
    params['gam2'] = gamma_2
    params['eps'] = eps_sq
    params['corrupt'] = corruption
    params['lcr'] = label_corruption_ratio
    params['lcs'] = label_corruption_seed
    params['tail'] = tail
    params['transform'] = transform
    params['save_dir'] = save_dir
    params['data_dir'] = data_dir
    params['pretrain_dir'] = pretrain_dir
    params['pretrain_epo'] = pretrain_epoch

    # initialize pipeline
    model_dir = os.path.join(save_dir,
                             f'sup_{arch}+{feature_dim}_{data}_epo{epochs}_bs{batch_size}_lr{learning_rate}_mom{momentum}_wd{weight_decay}_gam1{gamma_1}_gam2{gamma_2}_eps{eps_sq}_lcr{label_corruption_ratio}{tail}')
    utils.model.init_pipeline(model_dir=model_dir)

    # prepare for training
    if pretrain_dir is not None:
        net, _ = utils.model.load_checkpoint(pretrain_dir, pretrain_epoch)
        utils.model.update_params(model_dir, pretrain_dir)
    else:
        net = utils.model.load_architectures(arch, feature_dim)
    transforms = utils.dataset.load_transforms(transform)
    trainset = utils.dataset.load_trainset(data, transforms, path=data_dir)
    trainset = utils.corruption.corrupt_labels(corruption)(trainset, label_corruption_ratio, label_corruption_seed)
    trainloader = DataLoader(trainset, batch_size=batch_size, drop_last=True, num_workers=4)

    criterion = MaximalCodingRateReduction(eps=eps_sq, gamma_1=gamma_1, gamma_2=gamma_2)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [200, 400, 600], gamma=0.1)

    utils.model.save_params(model_dir, params)

    # train
    print("Starting training")
    
    for epoch in tqdm(range(epochs)):
        for step, (batch_imgs, batch_lbls) in enumerate(trainloader):
            features = net(batch_imgs.cuda())
            loss, loss_empi, loss_theo = criterion(features, batch_lbls, num_classes=trainset.num_classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            utils.model.save_state(model_dir, epoch, step, loss.item(), *loss_empi, *loss_theo)
        scheduler.step()
        if (epoch == epochs - 1) or (epoch % 100 == 0):
            utils.model.save_ckpt(model_dir, net, epoch)
    
    print("Training complete")


def self_supervised_train(params: dict) -> None:
    # (arch: str = 'resnet18', feature_dim: int = 128, data: str = 'cifar10', epochs: int = 800,
    #           batch_size: int = 1000, augmentations_per_mini_batch: int = 50, learning_rate: float = 0.001,
    #           momentum: float = 0.9,
    #           weight_decay: float = 5e-4, gamma_1: float = 1.,
    #           gamma_2: float = 1., eps_sq: float = 0.5, tail: str = '', transform: str = 'default',
    #           sampler: str = 'random',
    #           save_dir: str = './saved_models/',
    #           data_dir: str = './data/', pretrain_dir: str = None, pretrain_epoch: int = None)
    arch = params.get('arch', 'resnet18')
    feature_dim = params.get('fd', 32)
    data = params.get('data', 'cifar10')
    epochs = params.get('epo', 50)
    batch_size = params.get('bs', 1000)
    augmentations_per_mini_batch = params.get('aug', 50)
    learning_rate = params.get('lr', 0.001)
    momentum = params.get('mom', 0.9)
    weight_decay = params.get('wd', 5e-4)
    gamma_1 = params.get('gam1', 1.)
    gamma_2 = params.get('gam2', 10.)
    eps_sq = params.get('eps', 0.5)
    sampler = params.get('sampler', 'random')
    tail = params.get('tail', '')
    transform = params.get('transform', 'default')
    save_dir = params.get('save_dir', './saved_models/')
    data_dir = params.get('data_dir', './data/')
    pretrain_dir = params.get('pretrain_dir', None)
    pretrain_epoch = params.get('pretrain_epo', None)
    params['arch'] = arch
    params['fd'] = feature_dim
    params['data'] = data
    params['epo'] = epochs
    params['bs'] = batch_size
    params['aug'] = augmentations_per_mini_batch
    params['lr'] = learning_rate
    params['mom'] = momentum
    params['wd'] = weight_decay
    params['gam1'] = gamma_1
    params['gam2'] = gamma_2
    params['eps'] = eps_sq
    params['sampler'] = sampler
    params['tail'] = tail
    params['transform'] = transform
    params['save_dir'] = save_dir
    params['data_dir'] = data_dir
    params['pretrain_dir'] = pretrain_dir
    params['pretrain_epo'] = pretrain_epoch

    # initialize pipelines
    model_dir = os.path.join(save_dir,
                             f'selfsup_{arch}+{feature_dim}_{data}_epo{epochs}_bs{batch_size}_aug{augmentations_per_mini_batch}+{transform}_lr{learning_rate}_mom{momentum}_wd{weight_decay}_gam1{gamma_1}_gam2{gamma_2}_eps{eps_sq}{tail}')
    utils.model.init_pipeline(model_dir=model_dir)

    # prepare for training
    if pretrain_dir is not None:
        net, _ = utils.model.load_checkpoint(pretrain_dir, pretrain_epoch)
        utils.model.update_params(model_dir, pretrain_dir)
    else:
        net = utils.model.load_architectures(arch, feature_dim)
    transforms = utils.dataset.load_transforms(transform)
    trainset = utils.dataset.load_trainset(data, path=data_dir)
    trainloader = utils.augmentation.AugmentLoader(trainset,
                                                   transforms=transforms,
                                                   sampler=sampler,
                                                   batch_size=batch_size,
                                                   num_aug=augmentations_per_mini_batch)

    criterion = MaximalCodingRateReduction(eps=eps_sq, gamma_1=gamma_1, gamma_2=gamma_2)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60], gamma=0.1)

    utils.model.save_params(model_dir, params)

    # train
    print("Starting training")
    for epoch in tqdm(range(epochs)):
        for step, (batch_imgs, _, batch_idx) in enumerate(trainloader):
            batch_features = net(batch_imgs.cuda())
            loss, loss_empi, loss_theo = criterion(batch_features, batch_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            utils.model.save_state(model_dir, epoch, step, loss.item(), *loss_empi, *loss_theo)
            # if step % 20 == 0:
            #     utils.model.save_ckpt(model_dir, net, epoch)
        scheduler.step()
        if (epoch == epochs - 1) or (epoch % 100 == 0):
            utils.model.save_ckpt(model_dir, net, epoch)
    print("Training complete.")
