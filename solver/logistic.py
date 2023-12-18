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
def search(camid, bag, k):
    count = 0
    for i in bag:
        if (i==camid): 
            count+=1
        if (count==k):
            return False
    return True
def gen_factor(gallery_pid, gallery_feature, metric='cosine'):
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
        centers = torch.mean(gallery_feature[index_arr[1:]], 0)
        #print(gallery_feature[index_arr[0]].view(1, -1))
        point1 = F.normalize(
            gallery_feature[index_arr[0]].view(1, -1), p=2, dim=1)
        point2 = F.normalize(
            gallery_feature[index_arr[1]].view(1, -1), p=2, dim=1)
        centers = F.normalize(centers.view(1, -1), p=2, dim=1)
        if metric == 'cosine':
            cos_sim = float(torch.matmul(point1, point2.t())[0][0])
            cos_sim_c = float(torch.matmul(point1, centers.t())[0][0])
            positives.append([cos_sim, cos_sim_c])
        else:
            norm_sim = float(torch.cdist(point1, point2, p=2)[0][0])
            norm_sim_c = float(torch.cdist(point1, centers, p=2)[0][0])
            positives.append([norm_sim, norm_sim_c])

        # positives.append([norm_sim,norm_sim_c])
    for ii in tqdm(range(len(gallery_pid))):
        pid = np.unique(gallery_pid)
        m, n = np.random.choice(pid, 2)
        index_arr_m = np.where(gallery_pid == m)[0]
        index_arr_n = np.where(gallery_pid == n)[0]

        random.shuffle(index_arr_m)
        random.shuffle(index_arr_n)
        
        centers = torch.mean(gallery_feature[index_arr_n[1:]], 0)

        point1 = F.normalize(
            gallery_feature[index_arr_m[0]].view(1, -1), p=2, dim=1)
        point2 = F.normalize(
            gallery_feature[index_arr_n[0]].view(1, -1), p=2, dim=1)
        centers = F.normalize(centers.view(1, -1), p=2, dim=1)

        if metric == 'cosine':
            cos_sim = float(torch.matmul(point1, point2.t())[0][0])
            cos_sim_c = float(torch.matmul(point1, centers.t())[0][0])
            negatives.append([cos_sim, cos_sim_c])
        else:
            norm_sim = float(torch.cdist(point1, point2, p=2)[0][0])
            norm_sim_c = float(torch.cdist(point1, centers, p=2)[0][0])
            negatives.append([norm_sim, norm_sim_c])
    #     #negatives.append([norm_sim,norm_sim_c])


    print(len(positives), len(negatives))
    # positives = np.array(positives)
    # negatives = np.array(negatives)
    # np.save('/home/ceec/chuong/CLIP-ReID/datasets/positives.npy', positives)
    # np.save('/home/ceec/chuong/CLIP-ReID/datasets/negatives.npy', negatives)
    
    
    #if have file
    # positives = np.load('/home/ceec/chuong/CLIP-ReID/datasets/positives.npy') 
    # negatives = np.load('/home/ceec/chuong/CLIP-ReID/datasets/negatives.npy') 

    # # print(positives.shape, negatives.shape)
    Y = np.concatenate(
        (np.ones(len(positives)), np.zeros(len(negatives))), axis=0)
    X = np.concatenate((positives, negatives), axis=0)
    # Y_t=np.concatenate((np.ones(len(old_positives)), np.zeros(len(old_negatives))), axis=0)
    # X_t=np.concatenate((old_positives, old_negatives), axis=0)
    # scale = StandardScaler()
    # scaled_X = scale.fit_transform(X)
    

    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=100000, C=90)
    model = logistic_regression.fit(X, Y)
    print(model.score(X, Y), model.score(positives, np.ones(len(positives))),
          model.score(negatives, np.zeros(len(negatives))))
    print(model.coef_[0][0], model.coef_[0][1], model.intercept_[0])
    return model.coef_[0][0], model.coef_[0][1], model.intercept_

labels_gallery = np.load('/home/ceec/chuong/reid/baseline_train_person/pids.npy')

embeddings_gallery = torch.FloatTensor(np.load(
    '/home/ceec/chuong/reid/baseline_train_person/feats.npy'))
print(labels_gallery)
gen_factor(labels_gallery, embeddings_gallery)