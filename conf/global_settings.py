""" configurations for this project """


import os
from datetime import datetime

#cifar10 dataset path (python version)
#cifar10_PATH = '/nfs/private/cifar10/cifar-100-python'

#mean and std of cifar10 dataset
CIFAR10_TRAIN_MEAN = (0.49139968, 0.48215827 ,0.44653124)
CIFAR10_TRAIN_STD = (0.24703233 0.24348505 0.26158768)

#cifar10_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#cifar10_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 200
MILESTONES = [60, 120, 160]

#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
