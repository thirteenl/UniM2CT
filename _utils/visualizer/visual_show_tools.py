import os

import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

categories = {1: 'Replay', 2: 'Paper', 3: 'PartPGlass', 4: 'MaskTrans', 5: 'MaskPaper', 6: 'PartFunEye',
                      7: 'Ob', 8: 'MaskHalf', 9: 'Im', 10: 'Mann', 11: 'Co', 12: 'MaskSil', 13: 'PartEye',
                      14: 'StyleGAN', 15: 'F2F', 16: 'DeepFake', 17: 'FaceSwap', 18: 'StarGAN'}


class VisualShowTools(object):
    def __init__(self, save_path):
        self.save_path = save_path

    def draw_eer_curve(self, now_epoch, all_eer):
        plt.plot(now_epoch, all_eer, linewidth=2)
        plt.xlim([-0.5, len(now_epoch) + 10])
        plt.title('Cure of EER')
        plt.xlabel('Epoch')
        plt.ylabel('EER value (%)')
        plt.savefig(os.path.join(self.save_path, 'eer_curve.png'))
        plt.clf()

    def draw_tdr_curve(self, now_epoch, all_tdr):
        plt.plot(now_epoch, all_tdr, linewidth=2)
        plt.xlim([-0.5, len(now_epoch) + 10])
        plt.title('Cure of TDR')
        plt.xlabel('Epoch')
        plt.ylabel('TDR (0.2% FDR) value (%)')
        plt.savefig(os.path.join(self.save_path, 'tdr_curve.png'))
        plt.clf()

    def draw_auc_curve(self, now_epoch, all_auc):
        plt.plot(now_epoch, all_auc, linewidth=2)
        plt.xlim([-0.5, len(now_epoch) + 10])
        plt.title('Cure of AOC')
        plt.xlabel('Epoch')
        plt.ylabel('AOC value')
        plt.savefig(os.path.join(self.save_path, 'auc_curve.png'))
        plt.clf()

    def draw_boxplot(self, all_att_apcer):
        n_apcer = np.array(all_att_apcer)
        plt.boxplot(n_apcer)

        plt.title('APCER per Attack')
        plt.xlabel('Categories')
        plt.ylabel('APCER (%)')

        plt.xticks(range(1, len(categories) + 1), categories.values(), rotation=60, fontsize=6, ha='right', va='top')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'boxplot.png'), dpi=600)
        plt.clf()

    def draw_loss_curve(self, now_epoch, train_loss, test_loss):
        plt.plot(now_epoch, train_loss, label='train_loss', linewidth=2)
        plt.plot(now_epoch, test_loss, label='test_loss', linewidth=2)
        plt.xlim([-0.5, len(now_epoch) + 10])
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'loss_curve.png'))
        plt.clf()
