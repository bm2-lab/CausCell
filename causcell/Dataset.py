import numpy as np
import torch
from torch.utils import data
import joblib
import scanpy as sc
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, folder, profile_size, factor_list):
        super().__init__()
        self.folder = folder
        self.profile_size = profile_size
        # self.profile_data = sc.read_h5ad(folder).X
        
        # test model, we select profile_size features from data
        data = sc.read_h5ad(folder)
        self.profile_data = data.X[:, :profile_size]
        factor_dict=dict(zip(list(data.obs.columns), range(len(list(data.obs.columns)))))
        idx = [factor_dict[i] for i in factor_list]
        self.labels = data.obs.iloc[:,idx]
        
        tuple_list = []
        for idx in range(len(data)):
            tuple_list.append(str([data.obs[i][idx] for i in factor_list]))
        unique_tuplpe_list = np.unique(tuple_list)
        tmp_dict = dict(zip(unique_tuplpe_list, range(len(unique_tuplpe_list))))
        merged_class = [tmp_dict[i] for i in tuple_list]
        
        freq = {}
        for item in merged_class:
            if item in freq:
                freq[item] += 1
            else:
                freq[item] = 1
        
        for i in freq:
            freq[i] = 1 - (freq[i] / len(data))
        
        self.weights = np.array([freq[i] for i in merged_class])
        
    def __len__(self):
        return len(self.profile_data)
    
    def __getitem__(self, index):
        cell_exp = self.profile_data[index]
        labels = np.array(list(self.labels.iloc[index,:]), dtype=int)
        weight = self.weights[index]
        return cell_exp, labels, weight

class Generation_Dataset(data.Dataset):
    def __init__(self, folder, profile_size, factor_list):
        super().__init__()
        self.folder = folder
        self.profile_size = profile_size
        
        # test model, we select profile_size features from data
        data = sc.read_h5ad(folder)
        self.profile_data = data.X[:, :profile_size]
        factor_dict=dict(zip(list(data.obs.columns), range(len(list(data.obs.columns)))))
        idx = [factor_dict[i] for i in factor_list]
        self.labels = data.obs.iloc[:,idx]

    def __len__(self):
        return len(self.profile_data)
    
    def __getitem__(self, index):
        cell_exp = self.profile_data[index]
        return cell_exp