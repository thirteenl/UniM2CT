# -*- coding: utf-8 -*-
"""
@Time : 2024/3/22 9:43
@Auth : Thirteen
@File : FASTrain
@IDE : PyCharm
@Function : train the model 
"""
import os
import csv

import torch
from torch import nn
from abc import ABC

from trainer.BaseTrainer import BaseTrainer
from _utils.visualizer.visual_show_tools import VisualShowTools
from _utils.metrics_tools import BioMetricsTools


class FASTrainer(BaseTrainer, ABC):
    def __init__(self, cfg):
        super(FASTrainer, self).__init__(cfg)
        self.result = {'train_loss': [], 'val_loss': [], 'val_eer': [], 'val_tdr': [], 'val_auc': [], 'apcer': []}
        self.bmt = BioMetricsTools()

    def train(self, train_data_loader, **kwargs):
        if not train_data_loader:
            self.train_logger.warning("训练数据加载器为空")
            return

        self.model.train()
        batch_cost = 0

        for data in train_data_loader:
            img, label = data['image'].to(self.device), data['label'].to(self.device)
            self.optimizer.zero_grad()
            output = self.model(img)['logits'].squeeze(-1)

            loss = self.to_onehot_cal_loss(output, label)
            loss.backward()
            self.optimizer.step()
            batch_cost += loss.item()

        self.stepLR.step()
        self.result['train_loss'].append(batch_cost / len(train_data_loader))
        self.train_logger.info(
            '[Train] Epoch:[{}/{}]\t loss={:.5f}'.format(kwargs['epoch'] + 1, self.epochs,
                                                         batch_cost / len(train_data_loader)))

    def validate(self, val_data_loader, **kwargs):
        if not val_data_loader:
            self.val_logger.warning("验证数据加载器为空")
            return

        with torch.no_grad():
            self.model.eval()
            batch_cost = 0.0
            preds = None
            labels = None

            for data in val_data_loader:
                img, label = data['image'].to(self.device), data['label'].to(self.device)

                pred = self.model(img)['logits'].squeeze(-1)

                if preds is None:
                    preds = pred
                    labels = label
                else:
                    preds = torch.cat((preds, pred), dim=0)
                    labels = torch.cat((labels, label), dim=0)

            loss = self.to_onehot_cal_loss(preds, labels).item()

            # calculate metrics
            self.result['val_loss'].append(loss / len(val_data_loader))
            eer, auc, _, _ = self.bmt.get_eer(preds.detach().cpu().numpy(),
                                              labels.detach().cpu().numpy())
            self.val_logger.info(
                '[Validate] Epoch:[{}/{}]\t loss={:.5f}\t eer={:.2f}\t auc={:.4f}'.format(
                    kwargs['epoch'] + 1, self.epochs, batch_cost / len(val_data_loader), eer, auc))
            self.result['val_eer'].append(eer * 100)
            self.result['val_auc'].append(auc)
            self.draw_show_pic()

    def save_model(self, epoch):
        # save model of best result
        if len(self.result['val_eer']) > 0 and self.result['val_eer'][-1] == min(self.result['val_eer']):
            model_path = os.path.join(self.save_weights_folder, '{}-FAS-Best.pkl'.format(
                os.path.basename(self.save_folder)))
            torch.save(self.model.state_dict(), model_path)
            print(f"Save best model to {model_path}")

        if (epoch + 1) % 20 == 0:
            model_path = os.path.join(self.save_weights_folder,
                                      f'{os.path.basename(self.save_folder)}-FAS-{epoch + 1}.pkl')
            torch.save(self.model.state_dict(), model_path)
            print(f"Save model checkpoint at epoch {epoch + 1} done.")

    def save_result(self):
        # save eer and auc to csv file
        result_file = os.path.join(self.save_folder, 'val_result.csv')
        if not os.path.exists(os.path.dirname(result_file)):
            os.makedirs(os.path.dirname(result_file))
        with open(result_file, 'a', newline='') as csvfile:
            fieldnames = ['eer', 'auc']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not os.path.exists(os.path.join(self.save_folder, 'val_result.csv')):
                writer.writeheader()
            writer.writerow({'eer': self.result['val_eer'][-1], 'auc': self.result['val_auc'][-1]})

    def draw_show_pic(self):
        vst = VisualShowTools(self.save_folder)
        vst.draw_eer_curve([i for i in range(1, len(self.result['val_eer']) + 1)], all_eer=self.result['val_eer'])
        # vst.draw_tdr_curve([i for i in range(1, len(self.result['val_tdr']) + 1)], all_tdr=self.result['val_tdr'])
        vst.draw_auc_curve([i for i in range(1, len(self.result['val_auc']) + 1)], all_auc=self.result['val_auc'])

    def to_onehot_cal_loss(self, score, label):
        """live:0, spoof:1"""
        loss = self.loss_f(score, torch.where(label == 0, torch.zeros_like(label), torch.ones_like(label)))
        return loss
