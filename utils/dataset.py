import os

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# TODO: add training and testing dataset for anomaly detection.
# To reiterate, labels for testing dataset should be 0 (in distribution) vs 1 (out of distribution)
def load_trainset(name, transform=None, train=True, path="./data/"):
    """Loads a dataset for training and testing. If augmentloader is used, transform should be None.

    Parameters:
        name (str): name of the dataset
        transform (torchvision.transform): transform to be applied
        train (bool): load trainset or testset
        path (str): path to dataset base path

    Returns:
        dataset (torch.data.dataset)
    """
    _name = name.lower()
    if _name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(path, "cifar10"), train=train,
                                                download=True, transform=transform)
        trainset.num_classes = 10
    elif _name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(path, "cifar100"), train=train,
                                                 download=True, transform=transform)
        trainset.num_classes = 100
    elif _name == "cifar100coarse":
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(path, "cifar100"), train=train,
                                                 download=True, transform=transform)
        trainset.targets = sparse2coarse(trainset.targets)
        trainset.num_classes = 20
    elif _name == "mnist":
        trainset = torchvision.datasets.MNIST(root=os.path.join(path, "mnist"), train=train,
                                              download=True, transform=transform)
        trainset.num_classes = 10
    elif _name == "stl10":
        trainset = torchvision.datasets.STL10(root=os.path.join(path, "stl10"), split='train',
                                              transform=transform, download=True)
        testset = torchvision.datasets.STL10(root=os.path.join(path, "stl10"), split='test',
                                             transform=transform, download=True)
        trainset.num_classes = 10
        testset.num_classes = 10
        if not train:
            return testset
        else:
            trainset.data = np.concatenate([trainset.data, testset.data])
            trainset.labels = trainset.labels.tolist() + testset.labels.tolist()
            trainset.targets = trainset.labels
            return trainset
    elif _name == "stl10sup":
        trainset = torchvision.datasets.STL10(root=os.path.join(path, "stl10"), split='train',
                                              transform=transform, download=True)
        testset = torchvision.datasets.STL10(root=os.path.join(path, "stl10"), split='test',
                                             transform=transform, download=True)
        trainset.num_classes = 10
        testset.num_classes = 10
        if not train:
            return testset
        else:
            trainset.targets = trainset.labels
            return trainset
    else:
        raise NameError("{} not found in trainset loader".format(name))
    return trainset


def load_transforms(name):
    """Load data transformations.

    Note:
        - Gaussian Blur is defined at the bottom of this file.

    """
    _name = name.lower()
    if _name == "default":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    elif _name == "cifar":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()])
    elif _name == "mnist":
        transform = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomAffine((-90, 90)),
                transforms.RandomAffine(0, translate=(0.2, 0.4)),
                transforms.RandomAffine(0, scale=(0.8, 1.1)),
                transforms.RandomAffine(0, shear=(-20, 20))]),
            GaussianBlur(kernel_size=3),
            transforms.ToTensor()])
    elif _name == "stl10":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=9),
            transforms.ToTensor()])
    elif _name == "fashionmnist" or _name == "fmnist":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-90, 90)),
            transforms.RandomChoice([
                transforms.RandomAffine((-90, 90)),
                transforms.RandomAffine(0, translate=(0.2, 0.4)),
                transforms.RandomAffine(0, scale=(0.8, 1.1)),
                transforms.RandomAffine(0, shear=(-20, 20))]),
            GaussianBlur(kernel_size=3),
            transforms.ToTensor()])
    elif _name == "test":
        transform = transforms.ToTensor()
    else:
        raise NameError("{} not found in transform loader".format(name))
    return transform


def sort_dataset(data, labels, num_classes=10, stack=False):
    """Sort dataset based on classes.

    Parameters:
        data (np.ndarray): data array
        labels (np.ndarray): one dimensional array of class labels
        num_classes (int): number of classes
        stack (bol): combine sorted data into one numpy array

    Return:
        sorted data (np.ndarray), sorted_labels (np.ndarray)

    """
    sorted_data = [[] for _ in range(num_classes)]
    for i, lbl in enumerate(labels):
        sorted_data[lbl].append(data[i])
    sorted_data = [np.stack(class_data) for class_data in sorted_data]
    sorted_labels = [np.repeat(i, (len(sorted_data[i]))) for i in range(num_classes)]
    if stack:
        sorted_data = np.vstack(sorted_data)
        sorted_labels = np.hstack(sorted_labels)
    return sorted_data, sorted_labels


def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.

    Parameters:
        targets: torch.tensor(num_samples, ) vector of labels.

    Return:
        Pi: torch.tensor(num_classes, num_samples) -- membership matrix -- Pi[i, j] is sample j belongs to class i

    """
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = torch.tensor(np.zeros(shape=(num_classes, num_samples))).cuda()
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j] = 1.
    return Pi.float()


def membership_to_label(membership):
    """Turn a membership matrix into a list of labels."""
    _, num_classes, num_samples, _ = membership.shape
    labels = np.zeros(num_samples)
    for i in range(num_samples):
        labels[i] = np.argmax(membership[:, i])
    return labels


def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes. """
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot


## Additional Augmentations
class GaussianBlur():
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


def sparse2coarse(targets):
    """CIFAR100 Coarse Labels. """
    coarse_targets = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3,
                      9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10,
                      12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16,
                      4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12, 1,
                      9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6,
                      19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13]
    return np.array(coarse_targets)[targets]
