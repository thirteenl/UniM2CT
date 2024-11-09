# -*- coding: utf-8 -*-
"""
@Time : 2024/5/18 15:16
@Auth : Thirteen
@File : cfg.py
@IDE : PyCharm
@Function : 用于配置18种表征攻击的配置文件
"""
import inspect
import os
import time

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from dataset.FASDataSet import FASDataSet
from _utils.logging.logger import *

from models.UniM2CT import create_model

# +----------标签与所属类别----------------+
attack_type = {'live': [0],
               'physical_attack': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
               'digital_attack': [14, 15, 16, 17, 18],
               'FF++': [15, 16, 17],
               'GAN': [14, 18]
               }

Epochs = 60

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # cuda or cpu

# data file path
root_path = '/mnt/data'
train_txt_path = 'data/part_train.txt'
val_txt_path = 'data/part_val.txt'
test_txt_path = 'data/part_test.txt'

# dataset
train_dataset = DataLoader(FASDataSet(root_path, train_txt_path), batch_size=32, shuffle=True, num_workers=8)
val_dataset = DataLoader(FASDataSet(root_path, val_txt_path), batch_size=32, shuffle=True, num_workers=8)
test_dataset = DataLoader(FASDataSet(root_path, test_txt_path), batch_size=32, shuffle=True, num_workers=8)

'+------------- model --------------+'
model = create_model().to(device)

'+----------------------------------------------+'

loss_func = nn.BCELoss()  # binary cross entropy loss

optimizer = optim.AdamW(model.parameters(), lr=1e-5)  # optimizer
stepLR = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)  # scheduler

# result save folder
save_folder = 'result/{}-{}'.format("UniM2CT", time.strftime("%Y-%m-%d"))
filename = os.path.basename(save_folder)
save_weights_folder = os.path.join(save_folder, 'save_weights')
net_structure = inspect.getfile(create_model)  # 获取网络结构所在的文件名

# logging file
logger = get_logger
