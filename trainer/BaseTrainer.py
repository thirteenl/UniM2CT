# -*- coding: utf-8 -*-
"""
@Time : 2024/3/17 11:38
@Auth : Thirteen
@File : BaseTrainer
@IDE : PyCharm
@Function : base trainer
"""
import abc
import os
import sys
import torch
import shutil

from _utils.logging.logger import get_logger
from tqdm import tqdm


class BaseTrainer(object):

    def __init__(self, cfg):
        # init proper
        self.model = cfg.model
        self.epochs = cfg.Epochs
        self.optimizer = cfg.optimizer
        self.loss_f = cfg.loss_func
        self.device = cfg.device
        self.stepLR = cfg.stepLR
        self.save_folder = cfg.save_folder
        self.save_weights_folder = cfg.save_weights_folder
        self.filename = cfg.filename
        self.net_struc: str = cfg.net_structure
        self.logger = cfg.logger
        self.is_train = True

        self.train_logger = None
        self.val_logger = None

        # result set
        self.result = None

        # init related folder and logger
        self.init_folder()
        self.init_logger()

    def init_folder(self):
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
        if not os.path.exists(self.save_weights_folder):
            os.mkdir(self.save_weights_folder)

    def init_logger(self):
        self.train_logger = get_logger(os.path.join(self.save_folder, 'train.log'), target_name='train')
        self.val_logger = get_logger(os.path.join(self.save_folder, 'validate.log'), target_name='validate')
        # copy cfgs file to save folder
        shutil.copy('configs/cfg.py', os.path.join(self.save_folder, 'cfg.py'))
        try:
            shutil.copy(self.net_struc, os.path.join(self.save_folder, self.net_struc.split('/')[-1]))
        except FileNotFoundError:
            assert False, "path of network file is not exist,(your OS may be windows, please use linux)"

    @abc.abstractmethod
    def train(self, train_data, model, optimizer, loss_f, device, step_lr, **kwargs):
        """
        A training function that needs to be implemented by the user,
        and the training results are stored in 'self.result'.

        :param train_data: It is a batch of data loaded by 'dataloader'.
                        The specific format is determined by 'dataloader'
        :param train_data: It is a batch of data loaded by 'dataloader'.
        :param model: need to be trained model.
        :param optimizer:
        :param loss_f:
        :param device:
        :param step_lr:
        :param kwargs: other parameters.
        :return: None
        """
        raise NotImplementedError("train function is not implemented")

    @abc.abstractmethod
    def validate(self, val_data_loader, model, loss_f, device, **kwargs):
        """
        Test functions that need to be implemented by the user, with the training results stored in 'self.result'

        :param val_data_loader: It is a batch of data loaded by 'dataloader'.
                        The specific format is determined by 'dataloader'
        :param model: need to be tested model.
        :param loss_f: loss function.
        :param device:
        :param kwargs: other parameters.
        :return: None
        """
        raise NotImplementedError("test function is not implemented")

    def save_model(self, epoch):
        if (epoch + 1) % 20 == 0:
            model_path = os.path.join(self.save_weights_folder, '{}-FAS-{}.pkl'.format(
                os.path.basename(self.save_folder), str(epoch + 1)))
            torch.save(self.model.state_dict(), model_path)
            print("Save Modules Done !!!")

    def check_point(self, save_path):
        if os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path))
            print("Load Modules Done !!!")
        else:
            print("No such file or directory: {}".format(save_path))
