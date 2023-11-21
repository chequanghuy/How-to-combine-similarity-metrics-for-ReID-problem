import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
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

def gen_factor(gallery_pid, gallery_feature, gallery_base, g_camids, metric='cosine'):
    positives = []
    negatives = []
    #if do not have file
    for ii in tqdm(range(len(gallery_pid))):
        pid = np.unique(gallery_pid)
        m = np.random.choice(pid, 1)[0]
        index_arr = np.where(gallery_pid == m)[0]
        random.shuffle(index_arr)
        #print(index_arr)
        # print(gallery_feature[index_arr[1:]].shape)
        #print(gallery_feature[index_arr[0]].view(1, -1))
        point1 = gallery_feature[index_arr[0]].view(1, -1)
        point1_emd = gallery_base[index_arr[0]]
        indexdiff = np.where(g_camids != g_camids[index_arr[0]])[0]
        indexdiff_r = np.intersect1d(index_arr,indexdiff)
        point2 = gallery_feature[indexdiff_r[0]].view(1, -1)
        point2_emd = gallery_base[indexdiff_r[0]]
        if metric == 'cosine':
            cos_sim = emd_dis(point1.squeeze(0), None, point2, None, 0)
            cos_sim_c = emd_dis(point1_emd, None, point2_emd.unsqueeze(0), None, 1)
            positives.append([cos_sim, cos_sim_c])
        else:
            norm_sim = float(torch.cdist(point1, point2, p=2)[0][0])
            emd_sim = float(emd_dis(point1_emd, point2_emd))
            norm_sim_c = float(torch.cdist(point1, centers, p=2)[0][0])
            positives.append([norm_sim, emd_sim])
        # positives.append([norm_sim,norm_sim_c])
    for ii in tqdm(range(len(gallery_pid))):
        pid = np.unique(gallery_pid)
        m, n = np.random.choice(pid, 2)
        index_arr_m = np.where(gallery_pid == m)[0]
        index_arr_n = np.where(gallery_pid == n)[0]

        random.shuffle(index_arr_m)
        random.shuffle(index_arr_n)
        centers = torch.mean(gallery_feature[index_arr_n[1:]], 0)

        point1 = gallery_feature[index_arr_m[0]].view(1, -1)
        point1_emd = gallery_base[index_arr_m[0]]
        indexdiff = np.where(g_camids != g_camids[index_arr_m[0]])[0]
        indexdiff_r = np.intersect1d(index_arr_n,indexdiff)
        point2 = gallery_feature[indexdiff_r[0]].view(1, -1)
        point2_emd = gallery_base[indexdiff_r[0]]
        centers = centers.view(1, -1)
        if metric == 'cosine':
            cos_sim = emd_dis(point1.squeeze(0), None, point2, None,  0)
            cos_sim_c = emd_dis(point1_emd, None, point2_emd.unsqueeze(0), None, 1)
            negatives.append([cos_sim, cos_sim_c])
        else:
            norm_sim = float(torch.cdist(point1, point2, p=2)[0][0])
            emd_sim = float(emd_dis(point1_emd, point2_emd))
            norm_sim_c = float(torch.cdist(point1, centers, p=2)[0][0])
            negatives.append([norm_sim, emd_sim])
        #negatives.append([norm_sim,emd_dis])


    print(len(positives), len(negatives))
    positives = np.array(positives)
    negatives = np.array(negatives)
    np.save('logistic_data/positives_emdl3.npy', positives)
    np.save('logistic_data/negatives_emdl3.npy', negatives)
    
    
    #if have file
    # positives = np.load('logistic_data/positives_emdl3.npy', allow_pickle=True) 
    # negatives = np.load('logistic_data/negatives_emdl3.npy', allow_pickle=True) 

    # # print(positives.shape, negatives.shape)
    Y = np.concatenate(
        (np.ones(len(positives)), np.zeros(len(negatives))), axis=0)
    X = np.concatenate((positives, negatives), axis=0)
    # Y_t=np.concatenate((np.ones(len(old_positives)), np.zeros(len(old_negatives))), axis=0)
    # X_t=np.concatenate((old_positives, old_negatives), axis=0)
    # scale = StandardScaler()
    # scaled_X = scale.fit_transform(X)
    

    logistic_regression = LogisticRegression(class_weight='balanced',solver='lbfgs', max_iter=100000, C=1e5)
    model = logistic_regression.fit(X, Y)
    print(model.score(X, Y), model.score(positives, np.ones(len(positives))),
          model.score(negatives, np.zeros(len(negatives))))
    print(model.coef_[0][0], model.coef_[0][1], model.intercept_[0])
    return model.coef_[0][0], model.coef_[0][1], model.intercept_

q_camids = np.load('/home/ceec/chuong/reid-strong-baseline/rerank_person/q_camids_train.npy')
g_camids = np.load(
    '/home/ceec/chuong/reid-strong-baseline/rerank_person/g_camids_train.npy')
labels_query = np.load(
    '/home/ceec/chuong/reid-strong-baseline/rerank_person/q_pids_train.npy')

labels_gallery = np.load('/home/ceec/chuong/reid-strong-baseline/rerank_person/g_pids_train.npy')

embeddings_gallery = torch.FloatTensor(np.load(
    '/home/ceec/chuong/reid-strong-baseline/rerank_person/gf_train.npy'))
embeddings_query = torch.FloatTensor(np.load(
    '/home/ceec/chuong/reid-strong-baseline/rerank_person/qf_train.npy'))
q_base = torch.FloatTensor(np.load(
            '/home/ceec/chuong/reid-strong-baseline/rerank_person/q_layer3.npy'))
g_base = torch.FloatTensor(np.load(
    '/home/ceec/chuong/reid-strong-baseline/rerank_person/g_layer3.npy'))

gen_factor(labels_gallery, embeddings_gallery, g_base, g_camids)