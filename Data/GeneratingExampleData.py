import numpy as np
import scanpy as sc
import anndata as ad
import joblib
import scipy
import os

np.random.seed(888)

# generating an example training data
train_data_x = np.random.rand(2000, 1000).astype(np.float32)
concept_A = ["A","B","C","D"]
concept_A_array = np.random.choice(concept_A, 2000)
concept_B = ["q","w","e","r"]
concept_B_array = np.random.choice(concept_B, 2000)
concept_C = ['y','u','i']
concept_C_array = np.random.choice(concept_C, 2000)
train_data = ad.AnnData(X=train_data_x)
train_data.obs['concept_A'] = concept_A_array
train_data.obs['concept_B'] = concept_B_array
train_data.obs['concept_C'] = concept_C_array

# generating an example testing data
test_data_x = np.random.rand(500, 1000).astype(np.float32)
concept_A = ["A","B","C","D"]
concept_A_array = np.random.choice(concept_A, 500)
concept_B = ["q","w","e","r"]
concept_B_array = np.random.choice(concept_B, 500)
concept_C = ['y','u','i']
concept_C_array = np.random.choice(concept_C, 500)
test_data = ad.AnnData(X=test_data_x)
test_data.obs['concept_A'] = concept_A_array
test_data.obs['concept_B'] = concept_B_array
test_data.obs['concept_C'] = concept_C_array

# concept information
concept_list = ['concept_A','concept_B','concept_C']
concept_counts = [4, 4, 3]
concept_cdag = [[0,0,0,0],[0,0,0,0],[1,1,0,0],[0,0,0,0]]

if not os.path.exists("./Data"):
    os.makedirs("./Data")

train_data.write("./Data/example_train.h5ad")
test_data.write("./Data/example_test.h5ad")