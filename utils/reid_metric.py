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
from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking
from tqdm import tqdm
from tqdm.contrib import tzip
from .visrank import visualize_ranked_results
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

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
    def __init__(self, cfg, val_set, num_query, alpha, beta, max_rank=50, feat_norm='yes', metrics="", all_cameras = False, uncertainty=False, weighted=False, k=5, vis_top=0):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.metrics = metrics
        self.all_cameras = all_cameras
        self.uncertainty = uncertainty
        self.weighted = weighted
        self.k = k
        self.vis_top = vis_top
        self.cfg = cfg
        self.val_set = val_set
        self.dist_func = get_dist_func("cosine")
        self.alpha = alpha
        self.beta = beta

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
            print("Apply weighted K-nearest...")
        self.num_query = len(qf)
        camid2idx = defaultdict(list)
        index_g = []
        for idx, camid in enumerate(g_camids):
            index_g.append(idx)
            camid2idx[camid].append(idx)

        camid2idx_q = defaultdict(list)
        for idx, camid in enumerate(q_camids):
            camid2idx_q[camid].append(idx)
        unique_camids_q = sorted(np.unique(list(camid2idx_q.keys())))
        simmat = torch.tensor([])
        dict = {}

        #clf = SVC(kernel='poly', probability=True)
        # from sklearn.kernel_ridge import KernelRidge
        # X = np.load("/home/ceec/chuong/reid/X_train.npy")
        # Y = np.load("/home/ceec/chuong/reid/Y_train.npy")
        
        sim_gg = 0.8/(1+ torch.cdist(gf, gf, p=1)) +  0.55*torch.mm(gf, gf.t())
        for indx in tqdm(unique_camids_q):
            ind_qcamid = camid2idx[indx] # index of gallery feature has camid = q_camid

            ind_exq = np.setdiff1d(index_g, ind_qcamid) #index of gallery exp camid = q_camid
            gf_new = gf[ind_exq] # gallery faetureas with index = ind_exp
            sim_gg_exq = sim_gg[:, ind_exq] # similarity between galery features and gallery fetures exp q_camids
            sim_gg_argsort = np.argsort(1-sim_gg_exq, axis = 1) 
            sim_gg_argtopk = sim_gg_argsort[:, :k] #top_k index

            if weighted_knear:
                sim_gg_topk = sim_gg_exq[torch.arange(sim_gg_exq.size(0)).unsqueeze(1), sim_gg_argtopk] #top_k sim
                sum_sim_topk = torch.sum(sim_gg_topk, dim=-1)
                weight = sim_gg_topk /sum_sim_topk.view(-1, 1)
                gf_topk =  gf_new[sim_gg_argtopk].permute(0, 2, 1)
                centroid_g = torch.bmm(gf_topk, weight.unsqueeze(-1)).squeeze()

            else:
                gf_topk = gf_new[sim_gg_argtopk]
                sum_topk = torch.sum(gf_topk, dim = 1).squeeze()
                centroid_g = sum_topk/k 

            dict[indx] = centroid_g
        del sim_gg 
        for i in tqdm(range(len(q_ids))):

            curr_camid = q_camids[i] #camid of query
            sim_qg = torch.mm(qf[i].view(1, -1), gf.t())
            centroid_g = dict[curr_camid]
            sim_centroid = torch.mm(qf[i].view(1, -1), centroid_g.t())
        
            fusion = self.alpha*sim_qg + self.beta*sim_centroid #svm-linear
            #fusion = model.predict(np.column_stack((np.asarray(sim_qg)[0], np.asarray(sim_centroid)[0])))
            # fusion = model.predict(np.column_stack((np.asarray(sim_qg)[0], np.asarray(sim_centroid)[0])))
            #print(fusion[20])
            simmat = torch.cat((simmat, fusion.view(1, -1)), dim = 0)
        del dict
        return simmat
    

    def create_centroids_certainty(self, q_camids, g_camids, labels_query, labels_gallery, embeddings_query, embeddings_gallery, all_cameras=False):
        #print(embeddings_gallery[0])
        all_sim_qc = []
        for qf,q_id,q_camid in tzip(embeddings_query,labels_query,q_camids):
            centroids=[]
            cared={}
            for gf,g_id,g_camid in zip(embeddings_gallery,labels_gallery,g_camids):
                #if q_id == g_id:
                    # index_camid = np.where(g_camid != q_camid)[0]
                key = (q_id, g_id, q_camid)
                if key in cared.keys():
                    centroid = cared[key]
                else:
                    index_g = np.where(labels_gallery == g_id)[0]
                    index = [id for id in index_g if g_camids[id] != q_camid]
                    centroid = torch.mean(embeddings_gallery[index], 0)
                    cared[key] = centroid
                centroids.append(centroid)

            centroids = torch.stack(centroids)
            sim_qc = qf @ centroids.t()
            all_sim_qc.append(sim_qc.view(-1))
        all_sim_qc = torch.stack(all_sim_qc, dim=0)
        
        return all_sim_qc

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            print("The test feature is normalized")

        # feats = torch.FloatTensor(np.load("/home/ceec/chuong/reid/test_person/feats.npy"))
        # self.pids = np.load("/home/ceec/chuong/reid/test_person/f_ids.npy")
        # self.camids = np.load("/home/ceec/chuong/reid/test_person/f_camids.npy")
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
            print("Combining cosine and centroid as metric...")
            if self.uncertainty:
                print("Calculating centroid base on uncertainty...")
                # distmat = torch.FloatTensor(np.load(
                #     '/home/ceec/chuong/reid/dismat.npy'))
                simmat = \
                       self.create_centroids_uncertainty(q_camids, g_camids, q_pids, g_pids, qf, gf, k=self.k, all_cameras=self.all_cameras, weighted_knear=self.weighted)
            else:
                print("Calculating centroid base on certainty...")
                all_sim_qc = \
                        self.create_centroids_certainty(q_camids, g_camids, q_pids, g_pids, qf, gf, all_cameras=self.all_cameras)
                norm_sim = qf @ gf.t()
                simmat = self.alpha*norm_sim + self.beta*all_sim_qc   
            #np.save("/home/ceec/chuong/reid/simmat_weighted.npy", np.asarray(simmat.cpu()))

        elif (self.metrics == "centroid"):
            print("Using centroid as metric...")
            simmat = \
                self.create_centroids_certainty(q_camids, g_camids, q_pids, g_pids, qf, gf, all_cameras=self.all_cameras)

        elif (self.metrics == "cosine"):
            print("Using cosine as metric...")
            simmat = torch.mm(qf, gf.t())

        indices = torch.argsort(1-simmat, dim=1)
        del qf
        del gf
        cmc, mAP, _ = eval_func(np.asarray(indices.cpu(), dtype=np.int32), q_pids, g_pids, q_camids, g_camids,metrics=self.metrics, respect_camids=True)
        if self.vis_top:
            print("Start visualization...")
            visualize_ranked_results(
                1 - simmat,
                self.val_set,
                "image",
                self.cfg,
                width=self.cfg.INPUT.SIZE_TEST[1],
                height=self.cfg.INPUT.SIZE_TEST[0],
                save_dir=os.path.join(self.cfg.OUTPUT_DIR, "visrank"),
                topk=self.vis_top,
            )
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