import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from random import shuffle
import torch
from utils import *

import torch
import numpy as np
import torch.optim as optim
from torch import nn
import random
from model_structure import *

import os
import time
import copy
from argparse import ArgumentParser
# import sys
# sys.path.append('/home/ligl/project/GNet/GLPN')
# print(sys.path)

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_error(model, dataset_loader_test, device):

    time_dim = model.time_dimension
    error_MAE, error_RMSE = 0,0
    miss_num_all = 0
    for step, data_batch in enumerate(dataset_loader_test):

        node_mask = data_batch['node_mask']
        adj = data_batch['adj']
        adj_decoded = data_batch['adj_decoded']
        inputs = data_batch['features']
        mask_batch = data_batch['mask']
        rowsum = data_batch['rowsum']

        missing_mask = np.array(mask_batch)
        missing_inver =  ~mask_batch
        missing_index_s = 1 - missing_mask
        T_inputs = inputs * missing_index_s
        truth = inputs

        T_inputs = T_inputs.to(device)
        A_q = (calculate_random_walk_matrix_batch(adj).permute(0,2,1)).to(device)
        A_h = (calculate_random_walk_matrix_batch(adj.permute(0,2,1)).permute(0,2,1)).to(device)
        Adj = adj.to(device)

        imputation = model(T_inputs, A_q, A_h, Adj)
        # imputation = imputation.permute(0,2,1)
        imputation = imputation.cuda().data.cpu().numpy()
        o = imputation

        # truth = truth.permute(0,2,1)
        truth = truth.numpy()
        rowsum = rowsum.numpy()
        rowsum = rowsum.reshape(rowsum.shape[0],1,rowsum.shape[1])
        o = o * rowsum
        truth = truth * rowsum

        o[missing_inver == 1] = truth[missing_inver == 1]

        o = o * node_mask.numpy()
        truth = truth * node_mask.numpy()
        missing_mask = missing_mask * node_mask.numpy()

        error_MAE = error_MAE + np.sum(np.abs(o - truth))
        error_RMSE = error_RMSE + np.sum((o - truth) * (o - truth))
        miss_num_all = miss_num_all + np.sum(missing_mask)

    MAE = error_MAE / miss_num_all
    RMSE = np.sqrt(error_RMSE / miss_num_all)

    return MAE, RMSE, o, truth

def run(random_seed, args):
    graphs , max_nodes, feature_dim = Graph_load_batch(min_num_nodes=1, name=args.dataset)
    args.h = feature_dim
    num_graphs_raw = len(graphs)

    graphs_len = len(graphs)
    print('Number of graphs removed due to upper-limit of number of nodes: ',
          num_graphs_raw - graphs_len)
    graphs_test = graphs[:int(0.2 * graphs_len)]
    graphs_test_len = len(graphs_test)
    graphs_train = graphs[int(0.2 * graphs_len):]
    print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
    print('max number node: {}'.format(max_nodes))
    print('total graph num: {}, testing set: {}'.format(len(graphs), graphs_test_len))

    dataset = GraphAdjSampler(graphs_train, max_nodes, feature_dim)
    dataset_test = GraphAdjSampler(graphs_test, max_nodes, feature_dim)

    dataset_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=50,
            num_workers=0,
            shuffle=False)

    dataset_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=50,
            num_workers=0,
            shuffle=False)

    set_random_seed(random_seed)
    device = torch.device(args.device)
    save_path = "./result_best/k=%d_T=%d_Z=%d/%s/" % (args.K, args.h, args.z, args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_save_path = "./recode_model/"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    model = get_model(args, set = 'MultiGraph')
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
    scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2, last_epoch=-1)

    for epoch in range(args.epochs):
        time_s = time.time()
        losses = []
        for step, data_batch in enumerate(dataset_loader):
            node_mask = data_batch['node_mask']
            adj = data_batch['adj']
            adj_decoded = data_batch['adj_decoded']
            inputs = data_batch['features']
            mask_batch = data_batch['mask']

            missing_mask = np.array(mask_batch)
            missing_index = 1 - missing_mask
            Mf_inputs = inputs * missing_index
            Mf_inputs = Mf_inputs.to(device)
            mask = node_mask.to(device)  # The reconstruction errors on irregular 0s are not used for training
            # print('Mf_inputs.shape = ',Mf_inputs.shape)

            A_dynamic = adj  # Obtain the dynamic adjacent matrix
            A_q = (calculate_random_walk_matrix_batch(A_dynamic).permute(0,2,1)).to(device)
            A_h = (calculate_random_walk_matrix_batch(A_dynamic.permute(0,2,1)).permute(0,2,1)).to(device)
            Adj = A_dynamic.to(device)

            outputs = inputs.to(device)  # The label
            optimizer.zero_grad()
            X_res = model(Mf_inputs, A_q, A_h, Adj)

            loss = criterion(X_res * mask, outputs * mask)
            loss.backward()
            optimizer.step()  # Errors backward
            losses.append(loss.item())
        scheduler_lr.step()
        MAE_t, RMSE_t, pred, truth = test_error(model, dataset_loader_test, device)
        time_e = time.time()
        RMSE_list.append(RMSE_t)
        MAE_list.append(MAE_t)
        sum_loss = torch.sum(torch.tensor(losses))
        train_loss_list.append(sum_loss)
        print(random_seed, args.dataset, args.setting, args.miss_rate, args.model_name, epoch, MAE_t, RMSE_t,sum_loss,
              'time=',
              time_e - time_s)

        if MAE_t < best_mae:
            best_mae = MAE_t
            best_rmse = RMSE_t
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())

            # np.savez(save_path + args.model_name + str(
            #         int(args.miss_rate * 10)) + args.setting + '_' + "randomresult.npz", pred=pred,
            #          truth=truth)

    torch.save(best_model, 'recode_model/' + args.dataset + "_" + args.setting + "_" + args.model_name + str(
            random_seed) + '.pth')  # Save the model
    print("###############     best_result:        ")
    print("epoch = ", best_epoch, "     mae = ", best_mae, "     rmse = ", best_rmse)

    exname = args.model_name + args.setting + str(random_seed) + '_' + str(int(args.miss_rate * 10)) + '_'

    save_dir_base = './fig/' + args.dataset
    if not os.path.exists(save_dir_base):
        os.makedirs(save_dir_base)

    to_plot = True
    if to_plot == True:
        plt_result(args.dataset, exname, train_loss_list, RMSE_list, MAE_list, best_rmse, best_mae)

    print('dataset = ', args.dataset, 'exname = ', exname)
    return [best_epoch, best_mae, best_rmse]

def parse_args():

    parser = ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=-1)
    parser.add_argument("--model_name", type=str, default='GLPN')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--miss_rate', type=float, default=0.2)
    parser.add_argument('--p_obs', type=float, default=0.2)
    parser.add_argument("--dataset", type=str, default='FRANKENSTEIN') # Synthie, PROTEINS_full, FRANKENSTEIN

    parser.add_argument('--n_o_n_m', type=int, default=100)
    parser.add_argument('--n_m', type=int, default=50)
    parser.add_argument('--n_u', type=int, default=50)
    parser.add_argument('--h', type=int, default=18)
    parser.add_argument('--z', type=int, default=100)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--E_maxvalue', type=float, default=80)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--setting', type=str, default="MAR")
    parser.add_argument('--r', nargs='+', type=int,default = [50])

    args= parser.parse_args()
    print("dataset:{} h: {} z {} K {} lr {} E_maxvalue {} batch_size {} r {}".format(args.dataset, args.h, args.z,
                                                                                     args.K, args.lr,
                                                                                     args.E_maxvalue,
                                                                                     args.batch_size, args.r))
    return args

if __name__ == "__main__":
    random_seed_list = [0,32,64,128,256]
    run_results = []
    args = parse_args()
    for i in range(len(random_seed_list)):
        result = run(random_seed_list[i],args)
        run_results.append(result)

    for i in range(len(random_seed_list)):
        print(random_seed_list[i], "epoch = ", run_results[i][0], "   mae = ", run_results[i][1], "    rmse = ", run_results[i][2])

