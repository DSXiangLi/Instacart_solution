# -*- coding: utf-8 -*-

import logging


def logger_init(dt, name, console_only = True):
    # write log file for every day
    date_str = dt.strftime('%Y%m%d')

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s  - %(levelname)s - %(message)s")

    handler2 = logging.StreamHandler()
    handler2.setLevel(logging.INFO)
    handler2.setFormatter(formatter)

    if not console_only:
        handler1 = logging.FileHandler('./log/' + name + '_' + date_str + '.log')
        handler1.setLevel(logging.INFO)
        handler1.setFormatter(formatter)
        logger.addHandler(handler1)

    logger.addHandler(handler2)

    return logger
