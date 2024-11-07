# -*- coding: utf-8 -*-
"""
@Time : 2024/1/17 12:39
@Auth : Thirteen
@File : logger.py
@IDE : PyCharm
@Function : generate log file
"""
import logging


def get_logger(logger_name, verbosity=1, target_name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(target_name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(logger_name, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
