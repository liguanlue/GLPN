from __future__ import division

import torch
import numpy as np
from utils import *
import random

from fancyimpute import MatrixFactorization, IterativeImputer
from sklearn.neighbors import kneighbors_graph


random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
# cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Imputer:
    short_name: str

    def __init__(self, method=None, is_deterministic=True, in_sample=True):
        self.name = self.__class__.__name__
        self.method = method
        self.is_deterministic = is_deterministic
        self.in_sample = in_sample

    def fit(self, x, mask):
        if not self.in_sample:
            x_hat = np.where(mask, x, np.nan)
            return self.method.fit(x_hat)

    def predict(self, x, mask):
        x_hat = np.where(mask, x, np.nan)
        if self.in_sample:
            return self.method.fit_transform(x_hat)
        else:
            return self.method.transform(x_hat)

    def params(self):
        return dict()


class SpatialKNNImputer(Imputer):
    short_name = 'knn'

    def __init__(self, adj, k=20):
        super(SpatialKNNImputer, self).__init__()
        self.k = 2
        # normalize sim between [0, 1]
        sim = (adj + adj.min()) / (adj.max() + adj.min())
        knns = kneighbors_graph(1 - sim,
                                n_neighbors=self.k,
                                include_self=False,
                                metric='precomputed').toarray()
        self.knns = knns

    def fit(self, x, mask):
        pass

    def predict(self, x, mask):
        # x (5141, 207)
        x2=x.copy()
        x2[np.where(mask==0)]=np.nan
        global_mean = np.expand_dims(np.nanmean(x2, 0), 0)
        global_mean = np.repeat(global_mean, x2.shape[0], 0)
        X_com_global = x.copy()*mask + global_mean *(1-mask)
        x = np.where(mask, x, X_com_global)
        with np.errstate(divide='ignore', invalid='ignore'):
            print(x.shape)
            print(self.knns.shape)
            y_hat = (x @ self.knns.T) / (np.ones(mask.shape) @ self.knns.T)
        y_hat[~np.isfinite(y_hat)] = x.mean()
        return np.where(mask, x, y_hat)

    def params(self):
        return dict(k=self.k)

class MeanImputer(Imputer):
    short_name = 'mean'

    def fit(self, x, mask):
        d = np.where(mask, x, np.nan)
        self.means = np.nanmean(d, axis=0, keepdims=True)

    def predict(self, x, mask):
        if self.in_sample:
            d = np.where(mask, x, np.nan)
            means = np.nanmean(d, axis=0, keepdims=True)
        else:
            means = self.means
        return np.where(mask, x, means)

def  load_data(dataset, miss_rate, imputer_name):

    capacity = []
    if dataset == 'metr':
        A, X = load_metr_la_rdata()
        X = X[:,0,:]
        h = 24
    elif dataset == 'nrel':
        A, X , files_info = load_nerl_data()
        #For Nrel, We only use 7:00am to 7:00pm as the target data, because otherwise the 0-values of periods without sunshine will greatly influence the results
        time_used_base = np.arange(84,228)
        time_used = np.array([])
        for i in range(365):
            time_used = np.concatenate((time_used,time_used_base + 24*12* i))
        X=X[:,time_used.astype(np.int)]
        capacities = np.array(files_info['capacity'])
        capacities = capacities.astype('float32')
        h = 16
    elif dataset == 'sedata':
        A, X = load_sedata()
        A = A.astype('float32')
        X = X.astype('float32')
        h = 24
    elif dataset == 'pems':
        X, A = load_pems_data()
        A = A.astype(np.float32)
        A[0, 0] = 1
        X = X.transpose()
        h = 16
    else:
        raise NotImplementedError('Please specify datasets from: metr, nrel, ushcn, sedata or pems')

    split_line1 = int(X.shape[1] * 0.7)
    split_line2 = int(X.shape[1] * 0.80)

    training_set = X[:, :split_line1]
    print('training_set', training_set.shape)
    test_set = X[:, split_line2:]  # split the training and test period

    mask = MCAR(X, p = miss_rate, random_state = 64) #(True if the value is missing).

    A_1 = A + A.T
    A_bool = np.zeros(A.shape)
    A_bool[np.where(A_1 > 0)] = 1

    if imputer_name == 'mean':
        imputer = MeanImputer(in_sample=True)
    elif imputer_name == 'knn':
        imputer = SpatialKNNImputer(adj=A_bool, k=2)

    MAE, RMSE = get_error(imputer, X=test_set, test_slice = mask[:, split_line2:] , h=h)

    return  MAE, RMSE

def get_error(imputer, X, test_slice,h):

    in_sample = True
    train_slice = ~test_slice
    if in_sample:
        x_train, mask_train = X, train_slice
        y_hat = imputer.predict(x_train.T, mask_train.T)

    o = y_hat.T # N，D

    Missing0 = True

    truth = X
    o[train_slice == 1] = truth[train_slice == 1]

    if Missing0 == True:
        test_slice[truth == 0] = 0
        o[truth == 0] = 0

    MAE = np.sum(np.abs(o - truth)) / np.sum(test_slice)
    RMSE = np.sqrt(np.sum((o - truth) * (o - truth)) / np.sum(test_slice))

    return MAE, RMSE

if __name__ == "__main__":
    """
    Model training
    """
    datasets = ['metr','nrel','pems','sedata']
    impute_str = ['mean','knn']
    rate_list = [0.1,0.2, 0.3,0.5,0.7,0.9]
    for dataset in datasets:
        for j in impute_str:
            for i in range(6):
                miss_rate = rate_list[i]
                MAE, RMSE = load_data(dataset, miss_rate = miss_rate,imputer_name = j)
                print(dataset, 'MCAR' , "{} rate = {}, MAE ={:.6}, RMSE ={:.6}".format(j,miss_rate,MAE, RMSE))

