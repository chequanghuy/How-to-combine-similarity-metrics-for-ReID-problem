# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from utils.logistic import Logistic

from torch import nn

def create_supervised_logistic(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
    

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, ids = batch
            #print(torch.cuda.device_count())
            #data = data.to(device) #if torch.cuda.device_count() >= 1 else data
            data = data.to(device)
            feat = model(data)
            #print("Loading")
            return feat, ids

    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)
        qbar = ProgressBar()
        qbar.attach(engine)

    return engine


def do_train_logistic(
        cfg,
        metric,
        model,
        train_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    print("Create trainer for logistic")
    evaluator = create_supervised_logistic(model, metrics={'Logistic': Logistic(num_query, feat_norm=cfg.TEST.FEAT_NORM, metrics=metric)},
                                            device=device)
    evaluator.run(train_loader)
    score, score_pos, score_neg, alpha, beta = evaluator.state.metrics['Logistic']
    logger.info('Logistic Result:')
    logger.info("Score: {:.1%}".format(score))
    logger.info("Positive score: {:.1%}".format(score_pos))
    logger.info("Negative score: {:.1%}".format(score_neg))
    logger.info("Alpha value: {}".format(alpha))
    logger.info("Beta value: {}".format(beta))
