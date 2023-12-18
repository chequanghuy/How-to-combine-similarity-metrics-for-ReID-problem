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
from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking
import random
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression


def get_euclidean(x, y, **kwargs):
    m = x.shape[0]
    n = y.shape[0]
    distmat = (
        torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(x, y.t(),alpha=1, beta=-2)
    return distmat

def cosine_similarity(
    x_norm: torch.Tensor, y_norm: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Computes cosine similarity between two tensors.
    Value == 1 means the same vector
    Value == 0 means perpendicular vectors
    """
    # x_n, y_n = x.norm(dim=1)[:, None], y.norm(dim=1)[:, None]
    # x_norm = x / torch.max(x_n, eps * torch.ones_like(x_n))
    # y_norm = y / torch.max(y_n, eps * torch.ones_like(y_n))
    sim_mt = torch.mm(x_norm, y_norm.transpose(0, 1))
    return sim_mt


def get_cosine(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Computes cosine distance between two tensors.
    The cosine distance is the inverse cosine similarity
    -> cosine_distance = abs(-cosine_distance) to make it
    similar in behaviour to euclidean distance
    """
    sim_mt = cosine_similarity(x, y, eps)
    return torch.abs(1 - sim_mt).clamp(min=eps)


def get_dist_func(func_name="euclidean"):
    if func_name == "cosine":
        dist_func = get_cosine
    elif func_name == "euclidean":
        dist_func = get_euclidean
    #print(f"Using {func_name} as distance function during evaluation")
    return dist_func

class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes', metrics="", all_cameras = False, uncertainty=False, weighted=False, k=5):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.metrics = metrics
        self.all_cameras = all_cameras
        self.uncertainty = uncertainty
        self.weighted = weighted
        self.k = k
        self.dist_func = get_dist_func("cosine")
    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def create_centroids_uncertainty(self, q_camids, g_camids, q_ids, g_ids, qf, gf, k, all_cameras=False, weighted_knear=False):
        if weighted_knear:
            print("Apply weighted K-nearest")
        self.num_query = len(qf)
        camid2idx = defaultdict(list)
        for idx, camid in enumerate(g_camids):
            camid2idx[camid].append(idx)

        camid2idx_q = defaultdict(list)
        for idx, camid in enumerate(q_camids):
            camid2idx_q[camid] = idx
        unique_camids = sorted(np.unique(list(camid2idx.keys())))
        centroids_embeddings = []
        group_indices = torch.tensor([])
        for i in range(len(g_ids)):
            curr_camid = g_camids[i]
            used_camid = np.setdiff1d(unique_camids, curr_camid)
            sim = torch.mm(gf[i].view(1, -1), gf.t()).squeeze()
            # sim_exp_i = torch.cat((sim[:i], sim[i+1:]))
            group_indices = torch.tensor([])
            group_sim = torch.tensor([])
            if not all_cameras:
                check_camid = used_camid
            else:
                check_camid = unique_camids
            for camid in check_camid:
                inds = camid2idx[camid]
                # if not respect_camids:
                #     if (camid==curr_camid):
                #         inds = inds[inds != i]
                simmat_camid = sim[inds]
                inds = torch.tensor(inds).view(-1)
                simmat_sort = np.argsort(1-simmat_camid)
                simmat_sort_k = simmat_sort[:k]
                group_indices = torch.cat((group_indices, inds[simmat_sort_k]))
                if weighted_knear:
                    group_sim = torch.cat((group_sim, sim[inds[simmat_sort_k]]))
            group_indices = np.asarray(group_indices.cpu())
            
            if weighted_knear:
                sum = torch.sum(group_sim)
                weight = group_sim/sum
                centroids_embs = weight.view(-1, 1) * gf[group_indices]
                centroids_embs = torch.sum(centroids_embs, dim=0).view(1, -1)
            else:    
                centroids_embs = gf[group_indices]
                centroids_embs = self._calculate_centroids(centroids_embs, dim=0)

            centroids_embeddings.append(centroids_embs.detach().cpu())
        centroids_embeddings = torch.stack(centroids_embeddings).squeeze()
        return centroids_embeddings.cpu()
    

    def validation_create_centroids(self, q_camids, g_camids, labels_query, labels_gallery, embeddings_query, embeddings_gallery, all_cameras=False):
        #print(embeddings_gallery[0])
        self.num_query = len(embeddings_query)
        #print("num query", self.num_query)
        labels2idx = defaultdict(list)
        for idx, label in enumerate(labels_gallery):
            labels2idx[label].append(idx)

        labels2idx_q = defaultdict(list)
        for idx, label in enumerate(labels_query):
            labels2idx_q[label].append(idx)
        unique_labels = sorted(np.unique(list(labels2idx.keys())))
        labels = np.array(list(labels2idx.keys()))
        centroids_embeddings = []
        centroids_labels = []

        if not all_cameras:
            centroids_camids = []
            query_camid = q_camids

        # Create centroids for each pid seperately
        for label in unique_labels:
            cmaids_combinations = set()
            inds = labels2idx[label]
            inds_q = labels2idx_q[label]
            if not all_cameras:
                selected_camids_g = g_camids[inds]

                selected_camids_q = q_camids[inds_q]
                unique_camids = sorted(np.unique(selected_camids_q))

                for current_camid in unique_camids:
                    # We want to select all gallery images that comes from DIFFERENT cameraId
                    camid_inds = np.where(
                        selected_camids_g != current_camid)[0]
                    if camid_inds.shape[0] == 0:
                        continue
                    used_camids = sorted(
                        np.unique(
                            [cid for cid in selected_camids_g if cid != current_camid]
                        )
                    )
                    if tuple(used_camids) not in cmaids_combinations:
                        cmaids_combinations.add(tuple(used_camids))
                        centroids_emb = embeddings_gallery[inds][camid_inds]
                        centroids_emb = self._calculate_centroids(
                            centroids_emb, dim=0)
                        centroids_embeddings.append(
                            centroids_emb.detach().cpu())
                        centroids_camids.append(used_camids)
                        centroids_labels.append(label)

            else:
                centroids_labels.append(label)
                centroids_emb = embeddings_gallery[inds]
                centroids_emb = self._calculate_centroids(centroids_emb, dim=0)
                centroids_embeddings.append(centroids_emb.detach().cpu())
       
        centroids_embeddings = torch.stack(centroids_embeddings).squeeze()
        
        if not all_cameras:
            query_camid = [[item] for item in query_camid]
            centroids_camids = centroids_camids

        if all_cameras:

            camids_query = np.zeros_like(labels_query)
            camids_gallery = np.ones_like(np.array(centroids_labels))
            centroids_camids = np.hstack(
                (camids_query, np.array(camids_gallery)))
        
        
        centroids_embeddings = centroids_embeddings.cpu()

        centroids = torch.tensor([])
        for idx, label in enumerate(labels_gallery):
            indx_pid = np.where(centroids_labels == label)[0]
            if (indx_pid.size == 0):
                centroids = torch.cat((centroids, embeddings_gallery[idx].view(1, -1)), dim = 0)
                continue
            ind_new = []
            for i, centroid_camid in enumerate(centroids_camids):
                if (g_camids[idx] not in centroid_camid):
                    ind_new.append(i)
            index_final = np.intersect1d(indx_pid, ind_new)
            centroids = torch.cat((centroids, centroids_embeddings[index_final[0]].view(1, -1)), dim = 0)

        return centroids

    def _calculate_centroids(self,vecs, dim=1):
        # print(vecs.size())
        length = vecs.shape[dim]
        W=[]
        s = 0
        for i in range(length):
            s += (math.log(length+1)-math.log(i+1))
        for i in range(length):
            W.append((math.log(length+1)-math.log(i+1))/s)
        W=np.flip(np.array(W), axis=0).reshape(1,-1)
        W = torch.from_numpy(np.array(W)).type(torch.FloatTensor)
        centroid = torch.sum(vecs, dim).view(1, -1) / length
        return centroid
    
    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            print("The test feature is normalized")
        # query
        qf = feats[:self.num_query].cpu()
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:].cpu()
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        m, n = qf.shape[0], gf.shape[0]
        if (self.metrics == "cs+ct"):
            print("Combining cosine and centroid as metric")
            if self.uncertainty:
                print("Calculating centroid base on uncertainty")
                centroids_embeddings = \
                self.create_centroids_uncertainty(q_camids, g_camids, q_pids, g_pids, qf, gf, all_cameras=self.all_cameras, weighted_knear=self.weighted, k=self.k)
            else:
                print("Calculating centroid base on certainty")
                centroids_embeddings = \
                    self.validation_create_centroids(q_camids, g_camids, q_pids, g_pids, qf, gf, all_cameras=self.all_cameras)   
            
            norm_sim_c = torch.mm(qf, centroids_embeddings.t())
            norm_sim = torch.mm(qf, gf.t())
            distmat = -0.4160539651426207*norm_sim + 21.481786452879657*norm_sim_c #baseline market
            #distmat = 5.548447318974089*norm_sim + 13.11322659881579*norm_sim_c #baseline veri

        elif (self.metrics == "centroid"):
            print("Using centroid as metric")
            centroids_embeddings = \
                self.validation_create_centroids(q_camids, g_camids, q_pids, g_pids, qf, gf, all_cameras=self.all_cameras)
            distmat = torch.mm(qf, centroids_embeddings.t())

        else:
            print("Using cosine as metric")
            distmat = torch.mm(qf, gf.t())
        
        indices = torch.argsort(1-distmat, dim=1)
        cmc, mAP, _ = eval_func(np.asarray(indices.cpu(), dtype=np.int32), q_pids, g_pids, q_camids, g_camids,metrics=self.metrics, respect_camids=True)

        return cmc, mAP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
    def validation_create_centroids(self, q_camids, g_camids, labels_query, labels_gallery, embeddings_query, embeddings_gallery, respect_camids=True):
        #print(embeddings_gallery[0])
        self.num_query = len(embeddings_query)
        #print("num query", self.num_query)
        labels2idx = defaultdict(list)
        for idx, label in enumerate(labels_gallery):
            labels2idx[label].append(idx)

        labels2idx_q = defaultdict(list)
        for idx, label in enumerate(labels_query):
            labels2idx_q[label].append(idx)
        unique_labels = sorted(np.unique(list(labels2idx.keys())))
        labels = np.array(list(labels2idx.keys()))
        centroids_embeddings = []
        centroids_labels = []

        if respect_camids:
            centroids_camids = []
            query_camid = q_camids

        # Create centroids for each pid seperately
        for label in unique_labels:
            cmaids_combinations = set()
            inds = labels2idx[label]
            inds_q = labels2idx_q[label]
            if respect_camids:
                selected_camids_g = g_camids[inds]

                selected_camids_q = q_camids[inds_q]
                unique_camids = sorted(np.unique(selected_camids_q))

                for current_camid in unique_camids:
                    # We want to select all gallery images that comes from DIFFERENT cameraId
                    camid_inds = np.where(
                        selected_camids_g != current_camid)[0]
                    if camid_inds.shape[0] == 0:
                        continue
                    used_camids = sorted(
                        np.unique(
                            [cid for cid in selected_camids_g if cid != current_camid]
                        )
                    )
                    camid_unique = []
                    camid_bag = []
                    for index in camid_inds:
                        if (self.search(g_camids[index], camid_bag,5)==True):
                            camid_bag.append(g_camids[index])
                            camid_unique.append(index)
                        #g_camids[index] not in camid_bag or 
                    # print('---------', current_camid)
                    # print(g_camids[inds][camid_unique])
                    if tuple(used_camids) not in cmaids_combinations:
                        cmaids_combinations.add(tuple(used_camids))
                        centroids_emb = embeddings_gallery[inds][camid_inds]
                        centroids_emb = self._calculate_centroids(
                            centroids_emb, dim=0)
                        centroids_embeddings.append(
                            centroids_emb.detach().cpu())
                        centroids_camids.append(used_camids)
                        centroids_labels.append(label)

            else:
                # print("here", label)
                centroids_labels.append(label)
                centroids_emb = embeddings_gallery[inds]
                centroids_emb = self._calculate_centroids(centroids_emb, dim=0)
                centroids_embeddings.append(centroids_emb.detach().cpu())
        #print(centroids_labels)
        # Make a single tensor from query and gallery data
        centroids_embeddings = torch.stack(centroids_embeddings).squeeze()
        if respect_camids:
            query_camid = [[item] for item in query_camid]
            centroids_camids = centroids_camids

        if not respect_camids:

            camids_query = np.zeros_like(labels_query)
            camids_gallery = np.ones_like(np.array(centroids_labels))
            centroids_camids = np.hstack(
                (camids_query, np.array(camids_gallery)))
        return centroids_embeddings.cpu(), centroids_labels, centroids_camids
    
    def _calculate_centroids(self,vecs, dim=1):
        length = vecs.shape[dim]
        W=[]
        s = 0
        for i in range(length):
            s += (math.log(length+1)-math.log(i+1))
        for i in range(length):
            W.append((math.log(length+1)-math.log(i+1))/s)
        W=np.flip(np.array(W), axis=0).reshape(1,-1)
        W = torch.from_numpy(np.array(W)).type(torch.FloatTensor)
        centroid = torch.sum(vecs, dim).view(1, -1) / length
        return centroid
    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        ft_pids = np.asarray(self.pids)
        ft_camids = np.asarray(self.camids)
        # centroids_embeddings, centroids_labels, centroids_camids = \
        #     self.validation_create_centroids(q_camids, ft_camids, q_pids, ft_pids, qf, feats)
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        indices = np.argsort(distmat, axis=1)
        cmc, mAP, _ = eval_func(indices, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP