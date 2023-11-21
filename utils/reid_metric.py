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
def Sinkhorn(K, u, v):
    r = torch.ones_like(u) 
    c = torch.ones_like(v)
    thresh = 1e-1
    for _ in range(100):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    return T
def emd_dis(A,q, B,g, stage):
    if A is None or B is None:
        return 0.0
    # A(124, 16, 8, 1)
    # B(100, 124, 16, 8)
    #print(B.size())
    if (stage==0):
        sim = torch.einsum('c,nc->n', A, B)
    else:
        bs, N, x, y = B.size() 
        sim = torch.matmul(A.unsqueeze(-1).permute(1, 2, 3, 0), B.permute(2, 3, 1, 0)).view(bs, x, y)
        dis = 1.0 - sim
        K = torch.exp(-dis / 0.05)
        #K torch.Size([356, 16, 8])
        u = torch.zeros(bs, x, dtype=sim.dtype, device=sim.device).fill_(1. / x)
        v = torch.zeros(bs, y, dtype=sim.dtype, device=sim.device).fill_(1. / y)
        T = Sinkhorn(K, u, v)
        sim = torch.sum(T * sim, dim=(1,2))
        sim = torch.nan_to_num(sim) 
    return sim
def fusion_emd(qbase,qf, gbase, gf, k):
    emd_sim = torch.tensor([])
    dfusion = torch.tensor([])
    dismat = torch.tensor([])
    for idx in tqdm(range(qbase.size()[0])):
        anchor = qf[idx]
        sim = emd_dis(anchor,None, gf,None, 0) 
        sim_top = torch.argsort(sim, descending=True)
        top_inx = sim_top[:k]
        anchor_t = qbase[idx]
         #-30.521701038718994 -0.21232607035860396 37.907085005175624
         # 27.418632036397412 -4.21705050670176 -3.4351261750164537
         # layer3 28.01970077282917 -9.133969607028307 -2.670982981534271
        simavg = emd_dis(anchor_t, None, gbase[top_inx], None, 1)
        emd_sim = -4.21705050670176*simavg + 27.418632036397412*sim[top_inx] -3.4351261750164537
        rank_emd_sim = torch.argsort(emd_sim, descending=True)
        rank_real = top_inx[rank_emd_sim][:k]
        #print(rank_real)
        final_rank = torch.cat([rank_real, sim_top[k:]], dim=0)
        dismat = torch.cat((dismat, final_rank.view(1, -1)), dim=0)
        dfusion = torch.cat((dfusion, torch.tensor(emd_sim).view(1,-1)), dim=0)
    return dismat

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
    print(f"Using {func_name} as distance function during evaluation")
    return dist_func

class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes', metrics="", load_data=True):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.metrics = metrics
        self.load_data = load_data
        self.dist_func = get_dist_func("cosine")
    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.base_ = torch.tensor([])

    def update(self, output):
        feat, pid, camid, base = output
        if (self.load_data):
            self.feats.append(feat)
            #self.base_ = torch.cat((self.base_, base.cpu()), dim=0)
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
        # centroids_embeddings = torch.cat(
        #     (embeddings_query, centroids_embeddings), dim=0
        # )
        # centroids_labels = np.hstack(
        #     (labels_query, np.array(centroids_labels)))
        if respect_camids:
            query_camid = [[item] for item in query_camid]
            centroids_camids = centroids_camids

        if not respect_camids:

            camids_query = np.zeros_like(labels_query)
            camids_gallery = np.ones_like(np.array(centroids_labels))
            centroids_camids = np.hstack(
                (camids_query, np.array(camids_gallery)))
        
        # print("+++++++++++++++++++++++++")
        # print(centroids_embeddings.size())
        # #print(centroids_labels)
        # print(embeddings_query.size())
        return centroids_embeddings.cpu(), centroids_labels, centroids_camids
    
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
        # distmat = self.dist_func(x=centroid, y=vecs)
        # indices = np.argsort(distmat.cpu(), axis=1)
        # vecs = vecs[indices[0]]
        # atensor=torch.matmul(W,vecs)
        return centroid
    
    def compute(self):
        if (self.load_data):
            feats = torch.cat(self.feats, dim=0)
            print(feats.shape)
            #bases = self.base_
            if self.feat_norm == 'yes':
                feats = torch.nn.functional.normalize(feats, dim=1, p=2)
                #bases = torch.nn.functional.normalize(bases, dim=1, p=2)
                print("The test feature is normalized")
            # query
            qf = feats[:self.num_query]
            #q_base = bases[:self.num_query]
            q_pids = np.asarray(self.pids[:self.num_query])
            q_camids = np.asarray(self.camids[:self.num_query])
            # gallery
            gf = feats[self.num_query:]
            #g_base = bases[self.num_query:]
            g_pids = np.asarray(self.pids[self.num_query:])
            g_camids = np.asarray(self.camids[self.num_query:])
        else:
        # load data
            q_camids = np.load('/home/ceec/chuong/reid-strong-baseline/rerank_person/q_camids_train.npy')
            g_camids = np.load(
                '/home/ceec/chuong/reid-strong-baseline/rerank_person/g_camids_train.npy')
            q_pids = np.load(
                '/home/ceec/chuong/reid-strong-baseline/rerank_person/q_pids_train.npy')

            g_pids = np.load('/home/ceec/chuong/reid-strong-baseline/rerank_person/g_pids_train.npy')

            gf = torch.FloatTensor(np.load(
                '/home/ceec/chuong/reid-strong-baseline/rerank_person/gf_train.npy'))
            qf = torch.FloatTensor(np.load(
                '/home/ceec/chuong/reid-strong-baseline/rerank_person/qf_train.npy'))
            q_base = torch.FloatTensor(np.load(
                '/home/ceec/chuong/reid-strong-baseline/rerank_person/q_base.npy'))
            g_base = torch.FloatTensor(np.load(
                '/home/ceec/chuong/reid-strong-baseline/rerank_person/g_base.npy'))
        #   save to rank_person folder

        # np.save('rerank_person/q_layer3.npy', np.array(q_base.cpu()))
        # np.save('rerank_person/g_layer3.npy', np.array(g_base.cpu()))
        # print("Done Load")
        #return centroids_embeddings.cpu(), centroids_labels, centroids_camids
        # np.save('veri_person/qf_train.npy', np.array(qf.cpu()))
        # # torch.save('veri_persion/qf_train.pt', qf)
        # np.save('veri_person/q_pids_train.npy', np.array(q_pids))
        # np.save('veri_person/q_camids_train.npy', np.array(q_camids))
        # np.save('veri_person/gf_train.npy', np.array(gf.cpu()))
        # # torch.save('veri_persion/gf_train.pt', gf)
        # np.save('veri_person/g_pids_train.npy', np.array(g_pids))
        # np.save('veri_person/g_camids_train.npy', np.array(g_camids))
        m, n = qf.shape[0], gf.shape[0]
        if (self.metrics=="fusion_emd"):
            print("Combining cosine and EMD as metric")
            indices = fusion_emd(q_base, qf,  g_base, gf, 60)
            #-30.521701038718994 -0.21232607035860396 37.907085005175624
            #fusion = -30.28609228754813*norm_sim + -0.6022143629987687*emd_sim + 37.73627133742813
            #indices = np.argsort(dfusion, axis=1)
            cmc, mAP, all_topk = eval_func(np.asarray(indices, dtype=np.int32), q_pids, g_pids, q_camids, g_camids, metrics=self.metrics, respect_camids=True)

        elif (self.metrics == "fusion_centroid"):
            print("Combining cosine and centroid as metric")
            centroids_embeddings, centroids_labels, centroids_camids = \
                self.validation_create_centroids(q_camids, g_camids, q_pids, g_pids, qf, gf)
            centroids = torch.tensor([])
            for idx, label in enumerate(g_pids):
                indx_pid = np.where(centroids_labels == label)[0]
                if (indx_pid.size == 0):
                    centroids = torch.cat((centroids, gf[idx].view(1, -1)), dim = 0)
                    continue
                ind_new = []
                for i, centroid_camid in enumerate(centroids_camids):
                    if (g_camids[idx] not in centroid_camid):
                        ind_new.append(i)
                index_final = np.intersect1d(indx_pid, ind_new)
                centroids = torch.cat((centroids, centroids_embeddings[index_final[0]].view(1, -1)), dim = 0)    
            print(centroids.size())
            norm_sim_c = torch.mm(qf, centroids.t())
            norm_sim = torch.mm(qf, gf.t())
            fusion = -3.7351383911302336*norm_sim + -25.092212061307947*norm_sim_c + 30.12451975196364
            
            indices = np.argsort(fusion, axis=1)
            cmc, mAP, all_topk = eval_func(indices, q_pids, g_pids, q_camids, g_camids, metrics=self.metrics, respect_camids=True)
        elif (self.metrics == "centroid"):
            print("Using centroid as metric")
            centroids_embeddings, centroids_labels, centroids_camids = \
                self.validation_create_centroids(q_camids, g_camids, q_pids, g_pids, qf, gf)
            distmat = self.dist_func(x=qf, y=centroids_embeddings)
            indices = np.argsort(distmat, axis=1)
            cmc, mAP, _ = eval_func(indices, q_pids, np.asarray(centroids_labels), q_camids, np.asarray(centroids_camids), metrics=self.metrics, respect_camids=True)
        else:
            print("Using cosine as metric")
            norm_sim = torch.mm(qf, gf.t())
            indices = torch.argsort(norm_sim, dim=1, descending=True)
            cmc, mAP, all_topk = eval_func(np.asarray(indices.cpu(), dtype=np.int32), q_pids, g_pids, q_camids, g_camids,metrics=self.metrics, respect_camids=True)
        #market1501 centroid -3.7351383911302336 -25.092212061307947 30.12451975196364
        # market1501 emd  10.056792819033246 32.81131733666599 -19.98736619723309
        #fusion = -3.7351383911302336*norm_sim + -25.092212061307947*norm_sim_c + 30.12451975196364
        #fusion = 10.056792819033246*norm_sim + 32.81131733666599*emd_sim -19.98736619723309
        
        
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        #cmc, mAP = eval_func(centroids_embeddings, q_pids, g_pids, q_camids, g_camids)

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
        feat, pid, camid, _ = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print(qf.size())
        print("Enter reranking")
        # np.save('rerank_person/qf_train.npy', np.array(qf))
        # # torch.save('rerank_persion/qf_train.pt', qf)
        # np.save('rerank_person/q_pids_train.npy', np.array(q_pids))
        # np.save('rerank_person/q_camids_train.npy', np.array(q_camids))
        # np.save('rerank_person/gf_train.npy', np.array(gf))
        # # torch.save('rerank_persion/gf_train.pt', gf)
        # np.save('rerank_person/g_pids_train.npy', np.array(g_pids))
        # np.save('rerank_person/g_camids_train.npy', np.array(g_camids))
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        indices = np.argsort(distmat, axis=1)
        cmc, mAP, _ = eval_func(indices, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP