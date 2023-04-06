from __future__ import division

import torch
import numpy as np
import torch.optim as optim
from torch import nn
from utils import *
import random
from model_structure import *

import os
import time
import copy
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
    '''Load dataset
    Input: dataset name
    Returns
    -------
    A: adjacency matrix
    X: processed data
    capacity: only works for NREL, each station's capacity
    '''
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
    split_line2 = int(X.shape[1] * 0.7)

    training_set = X[:, :split_line1].transpose()
    print('training_set', training_set.shape)
    test_set = X[:, split_line2:].transpose()  # split the training and test period

    save_dir_base = './mask/' + args.dataset + args.setting + '/'
    if not os.path.exists(save_dir_base):
        os.makedirs(save_dir_base)

    if args.setting == "MCAR":
        mask = MCAR(X, p=args.miss_rate, random_state = random_seed)
    elif args.setting == "MAR":
        mask = MAR_mask(X.transpose(), p=args.miss_rate, p_obs=0.2)
    np.save(save_dir_base  + str(int(args.miss_rate * 10)) + "_mask.npy", mask)


    mask_train = mask[:, :split_line1].transpose()
    mask_test = mask[:, split_line2:].transpose()

    return A, X, training_set, test_set, mask_train, mask_test, capacities

"""
Define the test error
"""

def test_error(model, mask_test, test_set, A, Missing0, device,capacities):

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


def run(random_seed, args):

    set_random_seed(random_seed)
    device = torch.device("cuda:0")
    save_path = "./result_best/k=%d_T=%d_Z=%d/%s/" % (args.K, args.h, args.z, args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_save_path = "./recode_model/"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # load dataset
    A, X, training_set, test_set, mask_train, mask_test, capacities = load_data(args, random_seed)
    # Define model
    model = get_model(args)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    RMSE_list = []
    MAE_list = []
    pred = []
    truth = []
    print('##################################    start training    ##################################')
    best_mae = 100000
    train_loss_list = []
    scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5, last_epoch=-1)
    for epoch in range(args.epochs):
        time_s = time.time()
        losses = []
        for i in range(training_set.shape[0] // (args.h * args.batch_size)):  # using time_length as reference to record test_error
            t_random = np.random.randint(0, high=(training_set.shape[0] - args.h), size=args.batch_size, dtype='l')
            know_mask = set(random.sample(range(0, training_set.shape[1]), args.n_o_n_m))
            feed_batch = []
            mask_batch = []
            for j in range(args.batch_size):
                feed_batch.append(
                    training_set[t_random[j]: t_random[j] + args.h, :][:, list(know_mask)])  # generate 8 time batches
                mask_batch.append(mask_train[t_random[j]: t_random[j] + args.h, :][:, list(know_mask)])

            inputs = np.array(feed_batch)
            inputs_omask = np.ones(np.shape(inputs))
            if not args.dataset == 'nrel':
                inputs_omask[inputs == 0] = 0  # We found that there are irregular 0 values for METR-LA, so we treat those 0 values as missing data,
                # For other datasets, it is not necessary to mask 0 values

            missing_mask = np.array(mask_batch)
            missing_index = 1 - missing_mask

            if args.dataset == 'nrel':
                Mf_inputs = inputs * inputs_omask * missing_index
            else:
                Mf_inputs = inputs * inputs_omask * missing_index / args.E_maxvalue  # normalize the value according to experience
            Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32')).to(device)
            mask = torch.from_numpy(inputs_omask.astype('float32')).to(device)  # The reconstruction errors on irregular 0s are not used for training

            A_dynamic = A[list(know_mask), :][:, list(know_mask)]   # Obtain the dynamic adjacent matrix
            A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32')).to(device)
            A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32')).to(device)
            Adj = torch.from_numpy((A_dynamic).astype('float32')).to(device)

            if args.dataset == 'nrel':
                outputs = torch.from_numpy(inputs ).to(device)
            else:
                outputs = torch.from_numpy(inputs / args.E_maxvalue).to(device)  # The label

            optimizer.zero_grad()
            X_res = model(Mf_inputs, A_q, A_h, Adj)

            loss = criterion(X_res * mask, outputs * mask)
            loss.backward()
            optimizer.step()  # Errors backward
            losses.append(loss.item())
        scheduler_lr.step()
        if not args.dataset == 'nrel':
            MAE_t, RMSE_t, pred, truth = test_error(model, mask_test, test_set, A, True, device, capacities)
        else:
            MAE_t, RMSE_t, pred, truth = test_error(model, mask_test, test_set, A, False, device, capacities)

        time_e = time.time()
        RMSE_list.append(RMSE_t)
        MAE_list.append(MAE_t)
        train_loss_list.append(torch.sum(torch.tensor(losses)))
        print(random_seed, args.dataset, args.setting, args.miss_rate, args.model_name, epoch, MAE_t, RMSE_t, 'time=',
              time_e - time_s)

        if MAE_t < best_mae:
            best_mae = MAE_t
            best_rmse = RMSE_t
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())

            np.savez(save_path + args.model_name + str(int(args.miss_rate * 10)) + args.setting + '_' + "randomresult.npz", pred=pred, truth=truth)
        torch.save(best_model, 'recode_model/'+ args.dataset + "_"+ args.setting + "_" + args.model_name + str(args.miss_rate) +"_"+ str(random_seed)+'.pth')  # Save the model

    print("###############     best_result:        ")
    print("epoch = ", best_epoch, "     mae = ", best_mae, "     rmse = ", best_rmse)

    save_dir_base = './fig/' + args.dataset
    if not os.path.exists(save_dir_base):
        os.makedirs(save_dir_base)

    exname = args.model_name + args.setting + str(random_seed) + '_' + str(int(args.miss_rate * 10)) + '_'
    print('dataset = ',args.dataset, 'exname = ',exname)

    return [best_epoch, best_mae, best_rmse, ]

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
    random_seed_list = [0,32,64,128,256]
    run_results = []
    args = parse_args()
    for i in range(len(random_seed_list)):
        result = run(random_seed_list[i],args)
        run_results.append(result)
    print("###############     All_result:        ")
    for i in range(len(random_seed_list)):
        print(random_seed_list[i],  "   mae = ", run_results[i][1], "    rmse = ", run_results[i][2])




