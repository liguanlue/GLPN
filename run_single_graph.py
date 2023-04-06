from __future__ import division

import torch
import numpy as np
import torch.optim as optim
from torch import nn
from utils import *
import random
from model_structure import *
import sys
import os
import time
import copy
from argparse import ArgumentParser
from loggers import Logger, log
from datetime import datetime

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""
Define the test error
"""

def test_error(model, mask_test, test_set, A_all, Missing0, device,capacities, node_dim, rowsum):

    time_dim = model.time_dimension
    test_omask = np.ones(test_set.shape)
    test_inputs = (test_set * test_omask).astype('float32')
    test_inputs_s = test_inputs
    missing_index_s = 1 - mask_test
    o = np.zeros([test_set.shape[0] // time_dim * time_dim,test_inputs_s.shape[1]])  # Separate the test data into several h period

    for node_i in range(0, A_all.shape[0], node_dim):
        A = A_all[node_i:node_i + node_dim, :][:, node_i:node_i + node_dim]
        for i in range(0, test_set.shape[0] // time_dim * time_dim, time_dim):
            inputs = test_inputs_s[i:i + time_dim, :][:, node_i:node_i + node_dim]
            missing_inputs = missing_index_s[i:i + time_dim, :][:, node_i:node_i + node_dim]
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
            if torch.isinf(imputation).any() or torch.isnan(imputation).any():
                print('complex_names where inf encountered')
            imputation = imputation.data.cpu().numpy()
            o[i:i + time_dim, :][:, node_i:node_i + node_dim] = imputation[0, :, :]

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

    MAE = np.sum(np.abs(o - truth)) / np.sum(test_mask)
    RMSE = np.sqrt(np.sum((o - truth) * (o - truth)) / np.sum(test_mask))
    return MAE, RMSE, o, truth

def run(random_seed, args):
    set_random_seed(random_seed)
    device = torch.device(args.device)
    print(device)
    save_path = "./result_best/k=%d_T=%d_Z=%d/%s/" % (args.K, args.h, args.z, args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_save_path = "./recode_model/"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # load dataset
    A, X, training_set, test_set, mask_train, mask_test, capacities, rowsum = load_data_MCAR(args, random_seed)
    # Define model
    model = get_model(args)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    RMSE_list = []
    MAE_list = []
    print('##################################    start training    ##################################')
    best_mae = 100000
    train_loss_list = []
    scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5, last_epoch=-1)
    for epoch in range(args.epochs):
        time_s = time.time()
        losses = []
        model.train()
        for i in range(4):  # using time_length as reference to record test_error
            t_random = np.random.randint(0, high=(training_set.shape[0] - args.h), size=args.batch_size, dtype='l')
            know_mask = set(random.sample(range(0, training_set.shape[1]), args.n_o_n_m))
            feed_batch = []
            mask_batch = []
            for j in range(args.batch_size):
                feed_batch.append(
                    training_set[t_random[j]: t_random[j] + args.h, :][:,
                    np.array(list(know_mask))])  # generate 8 time batches
                mask_batch.append(mask_train[t_random[j]: t_random[j] + args.h, :][:, np.array(list(know_mask))])

            inputs = np.array(feed_batch)
            inputs_omask = np.ones(np.shape(inputs))
            if args.dataset in ['arxiv-year','yelp-chi']:
                inputs_omask[inputs == 0] = 0

            missing_mask = np.array(mask_batch)
            missing_index = 1 - missing_mask

            if args.dataset == 'nrel':
                Mf_inputs = inputs * inputs_omask * missing_index
            else:
                Mf_inputs = inputs * inputs_omask * missing_index / args.E_maxvalue  # normalize the value according to experience
            Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32')).to(device)
            mask = torch.from_numpy(inputs_omask.astype('float32')).to(
                device)  # The reconstruction errors on irregular 0s are not used for training

            A_dynamic = A[np.array(list(know_mask)), :][:,
                        np.array(list(know_mask))]  # Obtain the dynamic adjacent matrix
            A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32')).to(device)
            A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32')).to(device)
            Adj = torch.from_numpy((A_dynamic).astype('float32')).to(device)

            if args.dataset == 'nrel':
                outputs = torch.from_numpy(inputs).to(device)
            else:
                outputs = torch.from_numpy(inputs / args.E_maxvalue).to(device)  # The label
            optimizer.zero_grad()
            X_res = model(Mf_inputs, A_q, A_h, Adj)

            loss = criterion(X_res * mask, outputs * mask)
            loss.backward()
            optimizer.step()  # Errors backward
            losses.append(loss.item())
        scheduler_lr.step()

        model.eval()
        with torch.no_grad():
            if not args.dataset == 'nrel':
                MAE_t, RMSE_t, pred, truth = test_error(model, mask_test, test_set, A, args.missing0, device, capacities, args.node_dim, rowsum)
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

            torch.save(best_model, 'recode_model/' + args.dataset + "_" + args.setting + "_" + args.model_name + str(
                args.miss_rate) + "_" + str(random_seed) + '.pth')  # Save the model

    print("###############     best_result:        ")
    exname = args.model_name + args.setting + str(random_seed) + '_' + str(int(args.miss_rate * 10)) + '_'
    print('dataset = ', args.dataset, 'exname = ', exname)
    print("epoch = ", best_epoch, "     mae = ", best_mae, "     rmse = ", best_rmse)

    return [best_epoch, best_mae, best_rmse, ]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=-1)
    parser.add_argument("--model_name", type=str, default='GLPN')  # GLPN GCN_b
    parser.add_argument('--epochs', type=int, default = 200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--miss_rate', type=float, default=0.8)
    parser.add_argument('--p_obs', type=float, default=0.2)
    parser.add_argument("--dataset", type=str, default='cornell')  # cornell, texas,  Cora, CiteSeer, arxiv-year , yelp-chi
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--setting', type=str, default="MCAR")
    parser.add_argument('--missing0', action='store_true', default=True)

    args = parser.parse_args()

    if args.dataset == "Cora":  # 207
        total_n = 2708
        total_f = 1433
        args.r = [800]
        args.node_dim = 900
        args.n_u = 900
        args.n_o_n_m = 900
        args.n_m = math.ceil(args.n_o_n_m * args.miss_rate)
        args.h = 430
        args.z = 100
        args.K = 1
        args.lr = 0.008
        args.E_maxvalue = 1
        args.batch_size = 4
    elif args.dataset == "CiteSeer":  # 207
        total_n = 3327
        total_f = 3703
        args.r = [800]
        args.node_dim = 900
        args.n_u = 900
        args.n_o_n_m = 900
        args.n_m = math.ceil(args.n_o_n_m * args.miss_rate)
        args.h = 999
        args.z = 100
        args.K = 1
        args.lr = 0.001
        args.E_maxvalue = 1
        args.batch_size = 4
    elif args.dataset == 'cornell':
        total_n = 183
        total_f = 1703
        args.r = [170]
        args.node_dim = 183
        args.n_o_n_m = 183
        args.n_m = math.ceil(args.n_o_n_m * args.miss_rate)
        args.h = 511
        args.z = 100
        args.K = 1
        args.lr = 0.01   #0.007
        args.batch_size = 4
    elif args.dataset == 'texas':
        total_n = 183
        total_f = 1703
        args.r = [170]
        args.node_dim = 183
        args.n_o_n_m = 183
        args.n_m = math.ceil(args.n_o_n_m * args.miss_rate)
        args.h = 511
        args.z = 100
        args.K = 1
        args.lr = 0.005
        args.batch_size = 4
    elif args.dataset == 'arxiv-year':
        total_n = 169343
        total_f = 128
        args.r = [1400]
        args.node_dim = 1500
        args.n_o_n_m = 1500
        args.n_u = total_n - args.n_o_n_m
        args.n_m = math.ceil(args.n_o_n_m * args.miss_rate)
        args.h = 39
        args.z = 100
        args.K = 1
        args.lr = 0.0005
        args.batch_size = 4
    elif args.dataset == 'yelp-chi':
        total_n = 45954
        total_f = 32  # 9373
        args.r = [1800]
        args.node_dim = 2000  #
        args.n_o_n_m = 2000
        args.n_u = total_n - args.n_o_n_m
        args.n_m = math.ceil(args.n_o_n_m * args.miss_rate)
        args.h = 10
        args.z = 100
        args.K = 1
        args.lr = 0.001
        args.batch_size = 4

    else:
        raise NotImplementedError('Please specify datasets from: metr, nrel, sedata or pems')

    print("dataset:{} h: {} z {} K {} lr {}  batch_size {} r {}".format(args.dataset, args.h, args.z,
                                                                        args.K, args.lr, args.batch_size, args.r))
    return args

if __name__ == "__main__":
    random_seed_list = [0,32,64,128,256]
    run_results = []
    args = parse_args()
    start_time = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
    exname = args.dataset + args.model_name + args.setting +  '_' + str(int(args.miss_rate * 10)) + '_' + start_time
    if not os.path.exists('./log'):
        os.mkdir('./log')
    run_dir = os.path.join('log', exname)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    sys.stdout = Logger(logpath=os.path.join(run_dir, f'log.log'), syspart=sys.stdout)
    sys.stderr = Logger(logpath=os.path.join(run_dir, f'log.log'), syspart=sys.stderr)
    for i in range(len(random_seed_list)):
        result = run(random_seed_list[i], args)
        run_results.append(result)
    print("###############     All_result:        ")
    for i in range(len(random_seed_list)):
        print(random_seed_list[i], "   mae = ", run_results[i][1], "    rmse = ", run_results[i][2])




