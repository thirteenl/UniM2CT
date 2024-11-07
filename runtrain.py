# -*- coding: utf-8 -*-
"""
@Time : 2024/1/17 13:56
@Auth : Thirteen
@File : run
@IDE : PyCharm
@Function : run train and test file
"""
import time

from trainer.FASTrain import FASTrainer
from configs import cfg

main = FASTrainer(cfg)

if __name__ == '__main__':
    epochs = cfg.Epochs
    lr = 0
    print("Start Training -->", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for epoch in range(epochs):
        if lr != cfg.optimizer.state_dict()['param_groups'][0]['lr']:
            lr = cfg.optimizer.state_dict()['param_groups'][0]['lr']
            print("learning rate: {}".format(lr))

        main.train(cfg.train_dataset, epoch=epoch)
        main.validate(cfg.val_dataset, epoch=epoch)
        main.save_result()
        main.save_model(epoch)
