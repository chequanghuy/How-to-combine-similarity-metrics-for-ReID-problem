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

from utils.reid_metric import R1_mAP, R1_mAP_reranking
from utils.coeff_calc import Coeff
from torch import nn

def create_supervised_evaluator(model, metrics,
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
            data, pids, camids = batch
            #print(torch.cuda.device_count())
            #data = data.to(device) #if torch.cuda.device_count() >= 1 else data
            data = data.to(device)
            feat = model(data)
           
            #print("Loading")
            return feat, pids, camids

    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)
        qbar = ProgressBar()
        qbar.attach(engine)

    return engine

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


def inference(
        cfg,
        args,
        model,
        val_loader,
        train_loader,
        num_query,
        val_set
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    print("Create trainer for coeff...")

    evaluator1 = create_supervised_logistic(model, metrics={'Coeff': Coeff(num_query, feat_norm=cfg.TEST.FEAT_NORM, \
                                                    metrics=args.metric, method=args.coeff_method, n_data=args.n_data, rand_seed=args.rand_seed)},
                                            device=device)
    evaluator1.run(train_loader)
    score, score_pos, score_neg, alpha, beta = evaluator1.state.metrics['Coeff']

    del evaluator1
    # coeff = Coeff(num_query, feat_norm=cfg.TEST.FEAT_NORM, \
    #                                                 metrics=args.metric, method=args.coeff_method, n_data=args.n_data, rand_seed=args.rand_seed)
    # score, score_pos, score_neg, alpha, beta = coeff.compute()
    logger.info('Coefficent Result:')
    logger.info("Score: {:.1%}".format(score))
    logger.info("Alpha value: {}".format(alpha))
    logger.info("Beta value: {}".format(beta))

    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(cfg, val_set, num_query, alpha, beta, max_rank=50, \
                                feat_norm=cfg.TEST.FEAT_NORM, metrics=args.metric, all_cameras=args.all_cameras, \
                                    uncertainty=args.uncertainty, weighted=args.weighted, k=args.k, vis_top=args.vis_top)},
                                                device=device)
        # r1_map = R1_mAP(cfg, val_set, num_query, alpha, beta, max_rank=50, \
        #                         feat_norm=cfg.TEST.FEAT_NORM, metrics=args.metric, all_cameras=args.all_cameras, \
        #                             uncertainty=args.uncertainty, weighted=args.weighted, k=args.k, vis_top=args.vis_top)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))
    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    del evaluator
    
    
    # cmc, mAP = r1_map.compute()
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
