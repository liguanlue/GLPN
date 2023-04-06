from __future__ import division
import os
import zipfile
import numpy as np
import scipy.sparse as sp
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import scipy.io
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.utils import check_random_state
from scipy import optimize
import torch
import networkx as nx
from dataset import load_nc_dataset, load_newdata
from model_structure import *

def get_model(args, set = 'None'):
    if args.model_name == 'GLPN':
        model = GLPN(args.h, args.z, args.K, args.r, set)  # The graph neural networks
    if args.model_name == 'GCN_b':
        model = GCN_b(args.h, args.z, args.K)  # The graph neural networks
    return model

def load_data_MNAR(args, random_seed):

    capacities = []
    if args.dataset == 'metr':
        A, X = load_metr_la_rdata()
        X = X[:,0,:]
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
    elif args.dataset == 'cora':
        A, X = load_cora_data()
    elif args.dataset == 'CiteSeer':
        A, X = load_CiteSeer_data()
    elif args.dataset == 'pubmed':
        A, X = load_pubmed_data()
    elif args.dataset in ["cornell",'texas','wisconsin','chameleon', 'squirrel']:
        A, X, max_features = load_newdata(args.dataset)
        args.E_maxvalue = max_features
        A = A.astype('float32')
        X = X.astype('float32')
        print("{} max(x)={}".format(args.dataset, np.max(X)))
    elif args.dataset in ['wiki','arxiv-year', 'deezer-europe', 'Penn94', 'Genius']:
        A, X,  = load_nc_dataset(args.dataset)
        args.E_maxvalue = np.max(X)
    else:
        raise NotImplementedError('Please specify datasets from: metr, nrel, ushcn, sedata or pems')

    split_line1 = int(X.shape[1] * 0.7)

    training_set = X[:, :split_line1].transpose()
    print('training_set', training_set.shape)
    test_set = X[:, split_line1:].transpose()  # split the training and test period

    rand = np.random.RandomState(random_seed)  # Fixed random output
    unknow_set = rand.choice(list(range(0, X.shape[0])), args.n_u, replace=False)
    unknow_set = set(unknow_set)

    full_set = set(range(0, X.shape[0]))
    know_set = full_set - unknow_set
    training_set_s = training_set[:, np.array(list(know_set))]  # get the training data in the sample time period
    A_s = A[:, np.array(list(know_set))][np.array(list(know_set)), :]  # get the observed adjacent matrix from the full adjacent matrix,

    return A, X, training_set, test_set, unknow_set, full_set, know_set, training_set_s, A_s, capacities

def load_data_MCAR(args, random_seed):
    '''Load dataset
    Input: dataset name
    Returns
    -------
    A: adjacency matrix
    X: processed data
    capacity: only works for NREL, each station's capacity
    '''
    capacities = []
    rowsum = None
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
    elif args.dataset in ["cornell",'texas']:
        A, X, labels, rowsum = load_newdata(args.dataset)
        args.E_maxvalue = np.max(np.abs(X))
        print('E_maxvalue:', args.E_maxvalue)
        A = A.astype('float32')
        X = X.astype('float32')
        print("{} max(x)={}".format(args.dataset, np.max(X)))
    elif args.dataset in ['arxiv-year','yelp-chi']:
        A, X, labels = load_nc_dataset(args.dataset)
        args.E_maxvalue = np.max(np.abs(X))
    elif args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        A, X, labels, rowsum = load_nc_dataset(args.dataset)
        args.E_maxvalue = np.max(np.abs(X))
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

    return A, X, training_set, test_set, mask_train, mask_test, capacities, rowsum


def Graph_load_batch(min_num_nodes = 20, max_num_nodes = 1000, name = ' ',node_attributes = True,graph_labels=False):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    print('Loading graph dataset: '+str(name))
    G = nx.Graph()
    # load data
    path = './data/'+name+'/'
    data_adj = np.loadtxt(path+name+'_A.txt', delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(path+name+'_node_attributes.txt', delimiter=',')
    # data_node_label = np.loadtxt(path+name+'_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path+name+'_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path+name+'_graph_labels.txt', delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_att.shape[0]):
        if node_attributes:
            G.add_node(i+1, feature = data_node_att[i])
        # G.add_node(i+1, label = data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0])+1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator==i+1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['features'] = data_node_att[i]
        if G_sub.number_of_nodes()>=min_num_nodes and G_sub.number_of_nodes()<=max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
            # print(G_sub.number_of_nodes(), 'i', i)
    feature_dim = data_node_att.shape[1]
    print('Loaded', max_nodes, feature_dim)
    return graphs, max_nodes, feature_dim

def normalize_data(mx):
    """Row-normalize sparse matrix"""
    rowsum = mx.max(axis=1)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx, rowsum


class GraphAdjSampler(torch.utils.data.Dataset):
    def __init__(self, G_list,  max_nodes, feature_dim):
        self.max_num_nodes = max_nodes
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.rowsum_all = []
        self.node_mask_all =[]

        for G in G_list:
            adj = nx.to_numpy_array(G)
            # the diagonal entries are 1 since they denote node probability
            self.adj_all.append(
                np.asarray(adj) + np.identity(G.number_of_nodes()))
            self.len_all.append(G.number_of_nodes())
            features_batch = np.zeros([max_nodes, feature_dim])
            nodes_num = np.array(G.nodes).shape[0]
            node_mask = np.zeros([max_nodes, feature_dim])

            for k in range(nodes_num):
                a = G.nodes
                a = np.array(a)
                features_batch[k, :] = G.nodes[a[k]]['feature']
            features_batch, rowsum = normalize_data(features_batch)
            self.feature_all.append(features_batch.T)
            self.rowsum_all.append(rowsum)
            oneone = np.ones((nodes_num, feature_dim))
            node_mask[:nodes_num, :] = oneone
            self.node_mask_all.append(node_mask.T)

        self.mask = MCAR(self.feature_all[0].copy(), p=0.2, random_state=256)


    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        adj_decoded = np.zeros(self.max_num_nodes * (self.max_num_nodes + 1) // 2)
        node_idx = 0
        adj_vectorized = adj_padded[np.triu(np.ones((self.max_num_nodes, self.max_num_nodes))) == 1]


        return {'node_mask': self.node_mask_all[idx].copy(),
                'adj': adj_padded,
                'adj_decoded': adj_vectorized,
                'features': self.feature_all[idx].copy(),
                'rowsum': self.rowsum_all[idx].copy(),
                'mask': self.mask}


"""
Geographical information calculation
"""
def get_long_lat(sensor_index,loc = None):
    """
        Input the index out from 0-206 to access the longitude and latitude of the nodes
    """
    if loc is None:
        locations = pd.read_csv('data/metr/graph_sensor_locations.csv')
    else:
        locations = loc
    lng = locations['longitude'].loc[sensor_index]
    lat = locations['latitude'].loc[sensor_index]
    return lng.to_numpy(),lat.to_numpy()

def haversine(lon1, lat1, lon2, lat2): 
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r * 1000


"""
Load datasets
"""

def load_metr_la_rdata():
    if (not os.path.isfile("data/metr/adj_mat.npy")
            or not os.path.isfile("data/metr/node_values.npy")):
        with zipfile.ZipFile("data/metr/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall("data/metr/")

    A = np.load("data/metr/adj_mat.npy")
    X = np.load("data/metr/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)

    return A, X

def generate_nerl_data():
    # %% Obtain all the file names
    filepath = 'data/nrel/al-pv-2006'
    files = os.listdir(filepath)

    # %% Begin parse the file names and store them in a pandas Dataframe
    tp = [] # Type
    lat = [] # Latitude
    lng =[] # Longitude
    yr = [] # Year
    pv_tp = [] # PV_type
    cap = [] # Capacity MW
    time_itv = [] # Time interval
    file_names =[]
    for _file in files:
        parse = _file.split('_')
        if parse[-2] == '5':
            tp.append(parse[0])
            lat.append(np.double(parse[1]))
            lng.append(np.double(parse[2]))
            yr.append(np.int(parse[3]))
            pv_tp.append(parse[4])
            cap.append(np.int(parse[5].split('MW')[0]))
            time_itv.append(parse[6])
            file_names.append(_file)
        else:
            pass

    files_info = pd.DataFrame(
        np.array([tp,lat,lng,yr,pv_tp,cap,time_itv,file_names]).T,
        columns=['type','latitude','longitude','year','pv_type','capacity','time_interval','file_name']
    )
    # %% Read the time series into a numpy 2-D array with 137x105120 size
    X = np.zeros((len(files_info),365*24*12))
    for i in range(files_info.shape[0]):
        f = filepath + '/' + files_info['file_name'].loc[i]
        d = pd.read_csv(f)
        assert d.shape[0] == 365*24*12, 'Data missing!'
        X[i,:] = d['Power(MW)']
        print(i/files_info.shape[0]*100,'%')

    np.save('data/nrel/nerl_X.npy',X)
    files_info.to_pickle('data/nrel/nerl_file_infos.pkl')
    # %% Get the adjacency matrix based on the inverse of distance between two nodes
    A = np.zeros((files_info.shape[0],files_info.shape[0]))

    for i in range(files_info.shape[0]):
        for j in range(i+1,files_info.shape[0]):
            lng1 = lng[i]
            lng2 = lng[j]
            lat1 = lat[i]
            lat2 = lat[j]
            d = haversine(lng1,lat1,lng2,lat2)
            A[i,j] = d

    A = A/7500 # distance / 7.5 km
    A += A.T + np.diag(A.diagonal())
    A = np.exp(-A)
    np.save('data/nrel/nerl_A.npy',A)

def load_nerl_data():
    if (not os.path.isfile("data/nrel/nerl_X.npy")
            or not os.path.isfile("data/nrel/nerl_A.npy")):
        with zipfile.ZipFile("data/nrel/al-pv-2006.zip", 'r') as zip_ref:
            zip_ref.extractall("data/nrel/al-pv-2006")
        generate_nerl_data()
    X = np.load('data/nrel/nerl_X.npy')
    # A = np.load('data/nrel/nerl_A.npy')
    files_info = pd.read_pickle('data/nrel/nerl_file_infos.pkl')
    dist_mx = loadmat('data/nrel/nrel_dist_mx_lonlat.mat')
    dist_mx = dist_mx['nrel_dist_mx_lonlat']
    dis = dist_mx / 1e3
    A = np.exp(-0.5 * np.power(dis / 14, 2))

    X = X.astype(np.float32)
    # X = (X - X.mean())/X.std()
    return A,X,files_info

def load_sedata():
    assert os.path.isfile('data/sedata/A.mat')
    assert os.path.isfile('data/sedata/mat.csv')
    A_mat = scipy.io.loadmat('data/sedata/A.mat')
    A = A_mat['A']
    X = pd.read_csv('data/sedata/mat.csv',index_col=0)
    X = X.to_numpy()
    return A,X

def load_pems_data():
    assert os.path.isfile('data/pems/pems-bay.h5')
    assert os.path.isfile('data/pems/distances_bay_2017.csv')
    df = pd.read_hdf('data/pems/pems-bay.h5')
    # transfer_set = df.as_matrix()
    transfer_set = df.to_numpy(dtype=np.float32)
    distance_df = pd.read_csv('data/pems/distances_bay_2017.csv', dtype={'from': 'str', 'to': 'str'})
    normalized_k = 0.1

    dist_mx = np.zeros((325, 325), dtype=np.float32)

    dist_mx[:] = np.inf

    sensor_ids = df.columns.values.tolist()

    sensor_id_to_ind = {}

    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i
        
    for row in distance_df.values:
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                continue
            dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    adj_mx[adj_mx < normalized_k] = 0

    A_new = adj_mx
    return transfer_set,A_new
"""
Dynamically construct the adjacent matrix
"""

def MCAR(X, p, random_state):
    """
    Missing completely at random mechanism.
    Parameters
    ----------
    X : array-like, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have
        missing values.
    random_state: int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    mask : array-like, shape (n, d)
        Mask of generated missing values (True if the value is missing).
    """

    rng = check_random_state(random_state)

    n, d = X.shape
    mask = np.zeros((n, d))

    ber = rng.rand(n, d)
    mask = ber < p

    return mask

def MAR_mask(X, p, p_obs):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1) ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask.transpose()


def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs

def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts

def get_Laplace(A):
    """
    Returns the laplacian adjacency matrix. This is for C_GCN
    """
    if A[0, 0] == 1:
        A = A - np.diag(np.ones(A.shape[0], dtype=np.float32)) # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix. This is for K_GCN
    """
    if A[0, 0] == 0:
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32)) # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()

def calculate_random_walk_matrix_batch(adj):
    epis = 1e-15
    rowsum = torch.sum(adj, 1) + epis
    d_inv_sqrt = torch.pow(rowsum, -1)
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    random_walk_mx = torch.einsum("bmn,bnf->bmf", [adj, d_mat_inv_sqrt])
    return random_walk_mx

def plt_result(dataset, exname, train_loss_list,RMSE_list,MAE_list, best_rmse, best_mae):
    plt.figure(dpi=300)
    plt.plot(train_loss_list[5:])
    plt.title('Train Loss')
    plt.savefig('fig/' + dataset + '/' + exname + 'train_loss2.png')

    plt.figure(dpi=300)
    plt.plot(RMSE_list)
    plt.title('RMSE_list, Best RMSE:' + str(best_rmse))
    plt.savefig('fig/' + dataset + '/' + exname + 'RMSE_list2.png')

    plt.figure(dpi=300)
    plt.plot(MAE_list)
    plt.title('MAE_list, Best MAE:' + str(best_mae))
    plt.savefig('fig/' + dataset + '/' + exname + 'MAE_list2.png')


