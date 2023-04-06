from __future__ import division

import torch
import numpy as np
from utils import *
import random
from model_structure import *
import os

from argparse import ArgumentParser

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(args, random_seed):

    capacities = []
    if args.dataset == 'metr':
        A, X = load_metr_la_rdata()
        X = X[:,0,:]
        print("{} max(x)={}".format(args.dataset, np.max(X)))
    elif args.dataset == 'nrel':
        A, X , files_info = load_nerl_data()
        #For Nrel, We only use 7:00am to 7:00pm as the target data, because otherwise the 0-values of periods without sunshine will greatly influence the results
        time_used_base = np.arange(84,228)
        time_used = np.array([])
        for i in range(365):
            time_used = np.concatenate((time_used,time_used_base + 24*12* i))
        X=X[:,time_used.astype(np.int)]
        capacities = np.array(files_info['capacity'])
        capacities = capacities.astype('float32')
        X = (X.transpose()/capacities).transpose()
        print("{} max(x)={}".format(args.dataset, np.max(X)))
    elif args.dataset == 'pems':
        X, A = load_pems_data()
        A = A.astype(np.float32)
        A[0, 0] = 1
        X = X.transpose()
        print("{} max(x)={}".format(args.dataset, np.max(X)))
    elif args.dataset == 'sedata':
        A, X = load_sedata()
        A = A.astype('float32')
        X = X.astype('float32')
        print("{} max(x)={}".format(args.dataset, np.max(X)))

    else:
        raise NotImplementedError('Please specify datasets from: metr, nrel, sedata or pems')

    split_line1 = int(X.shape[1] * 0.7)
    split_line2 = int(X.shape[1] * 0.80)

    training_set = X[:, :split_line1].transpose()
    print('training_set', training_set.shape)
    test_set = X[:, split_line2:].transpose()  # split the training and test period

    mask = MCAR(X, p=args.miss_rate, random_state=random_seed)

    mask_train = mask[:, :split_line1].transpose()
    mask_test = mask[:, split_line2:].transpose()

    return A, X, training_set, test_set, mask_train, mask_test, capacities


"""
Define the test error
"""

def test_error(model, mask_test, test_set, A, Missing0, device,capacities, args):

    model.eval()
    with torch.no_grad():

        time_dim = model.time_dimension

        test_omask = np.ones(test_set.shape)
        if Missing0 == True:
            test_omask[test_set == 0] = 0
        test_inputs = (test_set * test_omask).astype('float32')
        test_inputs_s = test_inputs

        missing_index_s = 1 - mask_test

        o = np.zeros([test_set.shape[0] // time_dim * time_dim,
                      test_inputs_s.shape[1]])  # Separate the test data into several h period

        for i in range(0, test_set.shape[0] // time_dim * time_dim, time_dim):
            inputs = test_inputs_s[i:i + time_dim, :]
            missing_inputs = missing_index_s[i:i + time_dim, :]
            T_inputs = inputs * missing_inputs

            if args.dataset == 'nrel':
                T_inputs = T_inputs
            else:
                T_inputs = T_inputs / args.E_maxvalue

            T_inputs = np.expand_dims(T_inputs, axis=0)
            T_inputs = torch.from_numpy(T_inputs.astype('float32')).to(device)
            A_q = torch.from_numpy((calculate_random_walk_matrix(A).T).astype('float32')).to(device)
            A_h = torch.from_numpy((calculate_random_walk_matrix(A.T).T).astype('float32')).to(device)
            Adj = torch.from_numpy((A).astype('float32')).to(device)
            imputation = model(T_inputs, A_q, A_h, Adj)
            imputation = imputation.cuda().data.cpu().numpy()
            o[i:i + time_dim, :] = imputation[0, :, :]

        truth = test_inputs_s[0:test_set.shape[0] // time_dim * time_dim]
        if args.dataset == 'nrel':
            o = o
            truth = truth
            o = o * capacities
            truth = truth * capacities
        else:
            o = o * args.E_maxvalue

        o[missing_index_s[0:test_set.shape[0] // time_dim * time_dim] == 1] = truth[
            missing_index_s[0:test_set.shape[0] // time_dim * time_dim] == 1]

        test_mask = 1 - missing_index_s[0:test_set.shape[0] // time_dim * time_dim]
        if Missing0 == True:
            test_mask[truth == 0] = 0
            o[truth == 0] = 0

        MAE = np.sum(np.abs(o - truth)) / np.sum(test_mask)
        RMSE = np.sqrt(np.sum((o - truth) * (o - truth)) / np.sum(test_mask))

    return MAE, RMSE, o, truth

def parse_args():

    parser = ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=-1)
    parser.add_argument("--model_name", type=str, default='GLPN')
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--miss_rate', type=float, default=0.2)
    parser.add_argument('--p_obs', type=float, default=0.2)
    parser.add_argument("--dataset", type=str, default='metr') # metr, nrel, sedata or pems

    parser.add_argument('--n_o_n_m', type=int, default=100)
    parser.add_argument('--h', type=int, default=16)
    parser.add_argument('--z', type=int, default=100)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--E_maxvalue', type=float, default=80)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--setting', type=str, default="MCAR")
    parser.add_argument('--r', nargs='+', type=int,default = [100])

    args= parser.parse_args()

    if args.dataset == "metr": # 207
        args.n_o_n_m = 180
        args.h = 24
        args.z = 100
        args.K = 1
        args.lr = 0.001
        args.E_maxvalue = 70
        args.batch_size = 4
        args.r = [60]
    elif args.dataset == "nrel": # 137
        args.n_o_n_m = 100
        args.h = 16
        args.z = 100
        args.K = 1
        args.lr = 0.0015
        args.E_maxvalue = 80
        args.batch_size = 8
        args.r = [80]
    elif args.dataset == "pems": #325
        args.n_o_n_m = 250
        args.h = 16
        args.z = 100
        args.K = 1
        args.lr = 0.001
        args.E_maxvalue = 86  # 85.1
        args.batch_size = 10
        args.r = [80]
    elif args.dataset == "sedata": #323
        args.n_o_n_m = 280
        args.h = 24
        args.z = 100
        args.K = 1
        args.lr = 0.002
        args.E_maxvalue = 81  # 81.353516
        args.batch_size = 4
        args.r = [180]
    else:
        raise NotImplementedError('Please specify datasets from: metr, nrel, sedata or pems')

    print("dataset:{} h: {} z {} K {} lr {} E_maxvalue {} batch_size {} r {}".format(args.dataset,args.h, args.z,
                args.K,args.lr, args.E_maxvalue, args.batch_size, args.r))
    return  args


if __name__ == "__main__":

    random_seed = 0
    set_random_seed(random_seed)
    args = parse_args()
    device = torch.device("cuda:0")
    save_path = "./result_best/k=%d_T=%d_Z=%d/%s/" % (args.K, args.h, args.z, args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load dataset
    A, X, training_set, test_set, mask_train, mask_test, capacities = load_data(args, random_seed)
    # Define model
    model = GLPN(args.h, args.z, args.K, args.r)
    model.load_state_dict(torch.load('./recore_model/metr_MCAR_GLPN0.2_0.pth'))
    model.to(device)
    MAE, RMSE, pred, truth = test_error(model, mask_test, test_set, A, True, device, capacities, args)
    print("dataset = ", args.dataset ,"   mae = ",  MAE, "    rmse = ", RMSE)





