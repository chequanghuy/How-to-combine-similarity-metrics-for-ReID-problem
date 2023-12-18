# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import math
import numpy as np
import torch
from ignite.metrics import Metric
from collections import defaultdict
import torch.nn.functional as F
import random
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression



class Logistic(Metric):
    def __init__(self, num_query, feat_norm='yes', metrics=""):
        super(Logistic, self).__init__()
        self.num_query = num_query
        self.feat_norm = feat_norm
        self.metrics = metrics
    def reset(self):
        self.feats = []
        self.ids = []

    def update(self, output):
        feat, id = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(id))
    
    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            print("The train feature is normalized")
        f_ids = np.asarray(self.ids)
        metric = self.metrics
        positives = []
        negatives = []
        print("Load positive data:")
        for ii in tqdm(range(len(f_ids))):
            pid = np.unique(f_ids)
            m = np.random.choice(pid, 1)[0]
            index_arr = np.where(f_ids == m)[0]
            random.shuffle(index_arr)
            centers = torch.mean(feats[index_arr[1:]], 0)
            point1 = F.normalize(
                feats[index_arr[0]].view(1, -1), p=2, dim=1)
            point2 = F.normalize(
                feats[index_arr[1]].view(1, -1), p=2, dim=1)
            centers = F.normalize(centers.view(1, -1), p=2, dim=1)
            if metric == 'cosine':
                cos_sim = float(torch.matmul(point1, point2.t())[0][0])
                cos_sim_c = float(torch.matmul(point1, centers.t())[0][0])
                positives.append([cos_sim, cos_sim_c])
            else:
                norm_sim = float(torch.cdist(point1, point2, p=2)[0][0])
                norm_sim_c = float(torch.cdist(point1, centers, p=2)[0][0])
                positives.append([norm_sim, norm_sim_c])

        print("Load negative data:")
        for ii in tqdm(range(len(f_ids))):
            pid = np.unique(f_ids)
            m, n = np.random.choice(pid, 2)
            index_arr_m = np.where(f_ids == m)[0]
            index_arr_n = np.where(f_ids == n)[0]

            random.shuffle(index_arr_m)
            random.shuffle(index_arr_n)
            
            centers = torch.mean(feats[index_arr_n[1:]], 0)

            point1 = F.normalize(
                feats[index_arr_m[0]].view(1, -1), p=2, dim=1)
            point2 = F.normalize(
                feats[index_arr_n[0]].view(1, -1), p=2, dim=1)
            centers = F.normalize(centers.view(1, -1), p=2, dim=1)

            if metric == 'cosine':
                cos_sim = float(torch.matmul(point1, point2.t())[0][0])
                cos_sim_c = float(torch.matmul(point1, centers.t())[0][0])
                negatives.append([cos_sim, cos_sim_c])
            else:
                norm_sim = float(torch.cdist(point1, point2, p=2)[0][0])
                norm_sim_c = float(torch.cdist(point1, centers, p=2)[0][0])
                negatives.append([norm_sim, norm_sim_c])


        print('Length of positive data: ', len(positives))
        print('Length of positive data: ', len(negatives))
        Y = np.concatenate(
            (np.ones(len(positives)), np.zeros(len(negatives))), axis=0)
        X = np.concatenate((positives, negatives), axis=0)
        
        logistic_regression = LogisticRegression(solver='lbfgs', max_iter=10000)
        model = logistic_regression.fit(X, Y)
        score = model.score(X, Y)
        score_pos = model.score(positives, np.ones(len(positives)))
        score_neg = model.score(negatives, np.zeros(len(negatives)))
        alpha = model.coef_[0][0]
        beta = model.coef_[0][1]
      
        return score, score_pos, score_neg, alpha, beta