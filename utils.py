import random
import numpy as np
import os
import torch
import torch.nn as nn
from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix
import pandas as pd
import scipy.sparse as sp

def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False

# inf
def load_dataset(dataset, sens_attr, predict_attr, path, label_number):
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    print('dataset train test split: ',dataset) 
    header = list(idx_features_labels.columns) 
    header.remove(predict_attr) 
    header.remove('user_id')

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32) 
    labels = idx_features_labels[predict_attr].values 
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)} 

    features = torch.FloatTensor(np.array(features.todense())) 
    labels = torch.LongTensor(labels) 
    labels[labels >1] =1
    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0] 
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0) 
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], 
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], 
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])
    
    sens = idx_features_labels[sens_attr].values.astype(int) 
    sens = torch.FloatTensor(sens) 
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    # check 
    torch.save([features, labels, idx_train, idx_val, idx_test, sens],
                'data/inf/{}_information.pt'.format(dataset))

def build_relationship(feature, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(feature.T.T, feature.T.T)),
                              columns=feature.T.columns, index=feature.T.columns) 
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
