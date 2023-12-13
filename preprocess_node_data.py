import os
import scipy as sp
import numpy as np
import pandas as pd
import torch
from numpy import int64
import yaml
from utils import load_dataset
from scipy.spatial import distance_matrix
import time

def feature_normalize(feature):
    feature = np.array(feature)
    rowsum = feature.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return feature / rowsum


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns) #distance_matrix 计算任意两行之间的距离，第i行到第j行的距离返回在ij
    df_euclid = df_euclid.to_numpy()
    idx_map = [] 
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2] 
        neig_id = np.where(df_euclid[ind, :] > thresh*max_sim)[0] 
        import random
        random.seed(912)
        random.shuffle(neig_id) 
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig]) 
    idx_map =  np.array(idx_map)
    
    return idx_map



def generate_node_data(dataset, k_start, k_end, k_jump, sens_idex,self_loop=False):
    print('k_start:',k_start,'k_end:',k_end,'k_jump:',k_jump)
    start = time.time()
    if dataset in ['nba']:
    
        edges_unordered = np.genfromtxt('data/ori/nba/' + 'nba_relationship.txt').astype('int')
        node_df = pd.read_csv(os.path.join('data/ori/nba/', 'nba.csv'))
        
        print('load edge data')
        y = node_df["SALARY"].values
        labels = y
        adj_start = time.time()
        feature = node_df[node_df.columns[2:]]

        if sens_idex:
            feature = feature.drop(columns = ["country"])
        
        idx = node_df['user_id'].values # for relations
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
        adj = sp.sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) 
        if self_loop:
            adj = adj + sp.sparse.eye(adj.shape[0]) 
        else:
            print('no add self-loop')
        adj_end = time.time()
        print('create adj time is {:.3f}'.format((adj_end-adj_start)))
        # print('adj created!')
        feature = np.array(feature)
        feature = feature_normalize(feature)
        for i in range(k_start, k_end, k_jump):
            
            eigsh_start = time.time()
            eignvalue, eignvector = sp.sparse.linalg.eigsh(adj, which='LM', k=i)
            eigsh_end = time.time()
            print('eigsh time is {:.3f}'.format((eigsh_end-eigsh_start)))
            eignvalue = torch.FloatTensor(eignvalue)
            eignvector = torch.FloatTensor(eignvector)
            feature = torch.FloatTensor(feature)
            
            torch.save([eignvalue, eignvector, feature],  'data/eig/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')

    elif dataset in ['region_job']:
        edges_unordered = np.genfromtxt('data/ori/pokec/' +'region_job_relationship.txt').astype('int')
        print('load edge data')
        predict_attr = 'region'
        # ----
        print('Loading {} dataset'.format(dataset))

        idx_features_labels = pd.read_csv(os.path.join('data/ori/pokec/','region_job.csv'))
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(predict_attr)
        
        feature=feature_normalize(idx_features_labels[header])
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        #-----
        adj_start = time.time()
        if sens_idex:
            feature = feature.drop(columns = ["region"])

        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)} 
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
        adj = sp.sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if self_loop:
            adj = adj + sp.sparse.eye(adj.shape[0]) 
        else:
            print('no add self-loop')
        adj_end = time.time()
        print('create adj time is {:.3f}'.format((adj_end-adj_start)))
    
        for i in range(k_start, k_end, k_jump):
        
            eigsh_start = time.time()
            eignvalue, eignvector = sp.sparse.linalg.eigsh(adj, which='LM', k=i)
            eigsh_end = time.time()
            print('create eignvalue and eignvector')
            print('eigsh time is {:.3f}'.format((eigsh_end-eigsh_start)))
           
            eignvalue = torch.FloatTensor(eignvalue)
            eignvector = torch.FloatTensor(eignvector)
            feature = torch.FloatTensor(feature)

            torch.save([eignvalue, eignvector, feature],  'data/eig/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')
    elif dataset in ['region_job_2']:
        edge_df = pd.read_csv('data/ori/pokec/' + 'region_job_2_relationship.txt', sep='\t')
        node_df = pd.read_csv(os.path.join('data/ori/pokec/','region_job_2.csv')) 

        print('load edge data')
        y = node_df["I_am_working_in_field"].values
        adj_start = time.time()
        feature = node_df[node_df.columns[2:]]

        if sens_idex:
            feature = feature.drop(columns = ["region"])
        
        idx = node_df['user_id'].values # for relations
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
        adj = sp.sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) 
        if self_loop:
            adj = adj + sp.sparse.eye(adj.shape[0]) 
        else:
            print('no add self-loop')
        adj_end = time.time()
        print('create adj time is {:.3f}'.format((adj_end-adj_start)))
        feature = np.array(feature)
        feature = feature_normalize(feature)
        for i in range(k_start, k_end, k_jump):
            eigsh_start = time.time()
            eignvalue, eignvector = sp.sparse.linalg.eigsh(adj, which='LM', k=i)
            eigsh_end = time.time()
            print('create eignvalue and eignvector')
            print('eigsh time is {:.3f}'.format((eigsh_end-eigsh_start)))
            
            eignvalue = torch.FloatTensor(eignvalue)
            eignvector = torch.FloatTensor(eignvector)
            feature = torch.FloatTensor(feature)
            torch.save([eignvalue, eignvector, feature],  'data/eig/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')

    elif dataset in ['credit']:
        # dataset='credit'
        path='./data/ori/credit/'
        sens_attr="Age"
        predict_attr="NoDefaultNextMonth"
        # label_number=1000
        print('Loading {} dataset from {}'.format(dataset, path))
        idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset))) 
        header = list(idx_features_labels.columns) 
        header.remove(predict_attr) 
        header.remove('Single')
        # build relationship
        if os.path.exists(f'{path}/{dataset}_edges.txt'):
            edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int') 
        else:
            edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7) 
            np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered) 
        feature=feature_normalize(idx_features_labels[header])
        labels = idx_features_labels[predict_attr].values 
        
        adj_start = time.time()
        idx = np.arange(feature.shape[0]) 
        idx_map = {j: i for i, j in enumerate(idx)} 
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
        adj = sp.sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) 
        if self_loop:
            adj = adj + sp.sparse.eye(adj.shape[0]) 
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        print('create adj time is {:.3f}'.format((adj_end-adj_start)))
        for i in range(k_start, k_end, k_jump):
            eigsh_start = time.time()
            eignvalue, eignvector = sp.sparse.linalg.eigsh(adj, which='LM', k=i)
            eigsh_end = time.time()
            print('create eignvalue and eignvector')
            print('eigsh time is {:.3f}'.format((eigsh_end-eigsh_start)))
           
            eignvalue = torch.FloatTensor(eignvalue)
            eignvector = torch.FloatTensor(eignvector)
            feature = torch.FloatTensor(feature)

            torch.save([eignvalue, eignvector, feature],  'data/eig/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')

    elif dataset in ['income']:
        path='./data/ori/income/'
        sens_attr="race"
        predict_attr="income"
        print('Loading {} dataset from {}'.format(dataset, path))
        idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset))) 
        header = list(idx_features_labels.columns) 
        header.remove(predict_attr) 
       
        if os.path.exists(f'{path}/{dataset}_edges.txt'): 
            edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int') 
      
        feature=feature_normalize(idx_features_labels[header])
        labels = idx_features_labels[predict_attr].values 

        adj_start = time.time()
        idx = np.arange(feature.shape[0]) 
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) 
        adj = sp.sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) 
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) 
        if self_loop:
            adj = adj + sp.sparse.eye(adj.shape[0]) 
        adj_end = time.time()

        print('create adj time is {:.3f}'.format((adj_end-adj_start)))
        for i in range(k_start, k_end, k_jump):
            eigsh_start = time.time()
            eignvalue, eignvector = sp.sparse.linalg.eigsh(adj, which='LM', k=i)
            eigsh_end = time.time()
            print('create eignvalue and eignvector')
            print('eigsh time is {:.3f}'.format((eigsh_end-eigsh_start)))
           
            eignvalue = torch.FloatTensor(eignvalue)
            eignvector = torch.FloatTensor(eignvector)
            feature = torch.FloatTensor(feature)

            torch.save([eignvalue, eignvector, feature],  'data/eig_test/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')

    end = time.time()
    print('generate_node_data time cost is:{:.3f}'.format((end-start)))

if __name__ == '__main__':
    
    # pokec_z credit income
    datasets=['pokec_z']
    for i in datasets:

        config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)[i]  
        # generate_node_data(config['dataset'], 100, 200, 100, config['sens_idex'],self_loop=True)
        generate_node_data(config['dataset'], config['k_start'], config['k_end'], config['k_jump'], config['sens_idex'],self_loop=False)
        load_dataset(config['dataset'], config['sens_attr'], config['predict_attr'], config['path'],config['label_number'])
       