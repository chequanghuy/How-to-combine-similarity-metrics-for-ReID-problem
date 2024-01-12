# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir
import timm
import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger
from torch import nn
import random
import numpy as np

random.seed(1)
np.random.seed(1)



def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="configs/softmax_triplet_with_center.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--metric", type=str, help="Choose the metric in [\"cosine\", \"centroid\", \"cs+ct\"]", default="cosine")
    parser.add_argument('--all_cameras', action='store_true', help='Considering all cameras')
    parser.add_argument('--uncertainty', action='store_true', help=' Uncertain centroid calculation')
    parser.add_argument('--weighted', action='store_true', help='Use weighted centroid calculation when uncertainty is provided')
    parser.add_argument('--k', nargs='?', type=int, default=5, help='Top-k similarity base on uncertainty')
    parser.add_argument('--vis_top', nargs='?', type=int, default=0, help='Visualize top_k')
    parser.add_argument("--coeff_method", type=str, help="Choose the method to train in  [\"logistic\", \"svm\"]", default="logistic")
    parser.add_argument("--out", type=str, help="Save dir", default="output")
    parser.add_argument('--n_data', nargs='?', type=int, default=1000, help='Number of data to train to find coefficent')
    parser.add_argument('--rand_seed', nargs='?', type=int, default=0, help='random_seed')



    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = args.out
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes, val_set = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT)

    inference(cfg, args, model, val_loader, train_loader, num_query, val_set)


if __name__ == '__main__':
    main()
