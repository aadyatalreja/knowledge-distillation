""" helper functions """
import os
import sys
import re
import datetime

import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'kd_mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    else:
        print('Only vgg16 & mobilenet are supported')
        sys.exit()

    if args.gpu:  # use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of CIFAR10 training dataset
        std: std of CIFAR10 training dataset
        batch_size: dataloader batch size
        num_workers: dataloader num_workers
        shuffle: whether to shuffle
    Returns: train_data_loader: torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=False):
    """ return test dataloader
    Args:
        mean: mean of CIFAR10 test dataset
        std: std of CIFAR10 test dataset
        batch_size: dataloader batch size
        num_workers: dataloader num_workers
        shuffle: whether to shuffle
    Returns: cifar10_test_loader: torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader


def compute_mean_std(cifar10_dataset):
    """Compute the mean and std of CIFAR10 dataset
    Args:
        cifar10_training_dataset or cifar10_test_dataset
    Returns:
        tuple containing mean and std values of the dataset
    """

    data_r = np.dstack([cifar10_dataset[i][0][0, :, :] for i in range(len(cifar10_dataset))])
    data_g = np.dstack([cifar10_dataset[i][0][1, :, :] for i in range(len(cifar10_dataset))])
    data_b = np.dstack([cifar10_dataset[i][0][2, :, :] for i in range(len(cifar10_dataset))])
    
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std


class WarmUpLR(_LRScheduler):
    """Warm-up training learning rate scheduler
    Args:
        optimizer: optimizer (e.g. SGD)
        total_iters: total iters of warm-up phase
    """
    def init(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().init(optimizer, last_epoch)

    def get_lr(self):
        """First m batches set learning rate to base_lr * m / total_iters"""
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        Return most recent created folder under net_weights
        If no non-empty folder is found, return empty folder
    """
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f))) > 0]
    if len(folders) == 0:
        return ''

    # sort folders by creation time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


def most_recent_weights(weights_folder):
    """
        Return most recent created weights file
        If folder is empty, return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weight_files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]


def last_epoch(weights_folder):
    """Return last epoch from the weight file"""
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('No recent weights found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch


def best_acc_weights(weights_folder):
    """
        Return the best accuracy .pth file in the given folder
        If no best acc weights file is found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]