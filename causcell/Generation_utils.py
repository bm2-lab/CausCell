import torch
import numpy as np
import random
from .Modules import Denoise_net, GaussianDiffusion, Trainer
from .Dataset import Dataset, Generation_Dataset
import scanpy as sc
from torch.utils import data
import joblib
import os
import pandas as pd
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import anndata as ad
import itertools

def sampling(model, data_pwd, profile_size, factor_list, batch_size):
    with torch.no_grad():
        dataset = Generation_Dataset(data_pwd, profile_size, factor_list)
        dataloader = data.DataLoader(dataset, batch_size = batch_size, shuffle = False, pin_memory = True)
        disentanglement_embs = []
        for idx, data_ in enumerate(dataloader):
            batch_concept_embs = model.denosie_fn.DisentanglementEncoder.extract_exogenous_embs(data_.cuda()).cpu()
            disentanglement_embs.append(batch_concept_embs)
        concept_embs = [[] for _ in range(len(factor_list) + 1)]
        for factor in range(len(factor_list) + 1):
            for idx, i in enumerate(disentanglement_embs):
                for b in i:
                    concept_embs[factor].append(b[factor])
        
        concept_embs_stacked = []
        for i in concept_embs:
            concept_embs_stacked.append(np.array(torch.stack(i)))
    return np.array(concept_embs_stacked)

def extract_concept_embs(data_pwd, model, profile_size, factor_list, factor_counts, factor_cdag, batch_size):
    concept_embs = sampling(model, data_pwd, profile_size, factor_list, batch_size)
    for i in range(9):
        concept_embs += sampling(model, data_pwd, profile_size, factor_list, batch_size)
    concept_embs /= 10
    return concept_embs

def causality_based_concept_embs(exo_concept_embs, factor_cdag):
    z = np.matmul(np.linalg.inv(np.eye(len(factor_cdag)) - np.array(factor_cdag)), np.array(exo_concept_embs).transpose(1,0,2))
    return z.transpose(1,0,2).astype(np.float32)

def multi_target_generation(ori_data_pwd, 
                      training_data_pwd, 
                      train_concept_embs, 
                      model_save_pwd, 
                      factor_list, 
                      factor_counts, 
                      factor_cdag, 
                      multi_target_list, 
                      retain = True, 
                      name=''):
    ori_data = sc.read_h5ad(ori_data_pwd)
    unexplained_factor_pool = train_concept_embs[-1] 
    copyed_train_concept_embs = train_concept_embs.copy()
    for target_factor_dict in multi_target_list:
        target_factor = target_factor_dict["target_factor"]
        ref_factor_value = target_factor_dict["ref_factor_value"]
        tgt_factor_value = target_factor_dict["tgt_factor_value"]
        target_embeddings = train_concept_embs[factor_list.index(target_factor)]
        if ref_factor_value != 'all':
            ref_factor_embeddings = target_embeddings[ori_data.obs[target_factor]==ref_factor_value]
            tgt_factor_embeddings = target_embeddings[ori_data.obs[target_factor]==tgt_factor_value]
            sampled_indices = np.random.choice(tgt_factor_embeddings.shape[0], size=ref_factor_embeddings.shape[0], replace=True)
            generated_tcell_embs = tgt_factor_embeddings[sampled_indices]
            train_concept_embs[factor_list.index(target_factor)][ori_data.obs[target_factor]==ref_factor_value]=generated_tcell_embs
        else:
            ref_factor_embeddings = target_embeddings[ori_data.obs[target_factor]!=tgt_factor_value]
            tgt_factor_embeddings = target_embeddings[ori_data.obs[target_factor]==tgt_factor_value]
            sampled_indices = np.random.choice(tgt_factor_embeddings.shape[0], size=target_embeddings.shape[0], replace=True)
            generated_tcell_embs = tgt_factor_embeddings[sampled_indices]
            train_concept_embs[factor_list.index(target_factor)]=generated_tcell_embs

    training_output_pwd = model_save_pwd
    training_num_steps = 100000
    training_batch_size = 64
    if ori_data.X.shape[1] > 2000:
        training_profile_size = 2000
    else:
        training_profile_size = ori_data.X.shape[1]
    training_lr = 2e-5

    profile_size = training_profile_size
    model_denoise = Denoise_net(profile_size, profile_size, len(factor_counts) + 1, torch.tensor(factor_cdag).float(), factor_counts).cuda()
    GaussianDiffusion_model = GaussianDiffusion(model_denoise, profile_size = profile_size, timesteps = 1000).cuda()
    trainer = Trainer(GaussianDiffusion_model, training_data_pwd, factor_list,
                    profile_size = training_profile_size, 
                    train_batch_size = training_batch_size, 
                    train_lr = training_lr,
                    train_num_steps = training_num_steps, 
                    results_folder = training_output_pwd, 
                    train_log = False)
    trainer.load(10)
    train_concept_embs = np.array(train_concept_embs)
    if retain:
        selected_ids = np.array([True]*len(train_concept_embs[0]))
        for target_factor_dict in multi_target_list:
            target_factor = target_factor_dict["target_factor"]
            ref_factor_value = target_factor_dict["ref_factor_value"]
            tgt_factor_value = target_factor_dict["tgt_factor_value"]
            if ref_factor_value != "all":
                selected_ids = selected_ids & np.array((ori_data.obs[target_factor]==ref_factor_value))
            else:
                selected_ids = selected_ids & np.array((ori_data.obs[target_factor] != tgt_factor_value))
        train_concept_embs = train_concept_embs[:,selected_ids,:]
        ori_train_concept_embs = copyed_train_concept_embs[:,selected_ids,:]
        train_concept_embs = np.concatenate([train_concept_embs, ori_train_concept_embs], axis=1)

    with torch.no_grad():
        generated_samples = []
        batch_sz = 1024
        steps = train_concept_embs[0].shape[0]//1024 + 1
        for ii in range(steps):
            batch_concept_embs = train_concept_embs[:, ii*batch_sz:(ii+1)*batch_sz,:]
            batch_concept_embs = np.transpose(batch_concept_embs, (1,0,2))
            samples_with_cross_attention = trainer.model.sample_with_factor(concept_embs = torch.tensor(batch_concept_embs).cuda(), batch_size = batch_concept_embs.shape[0])
            generated_samples.append(samples_with_cross_attention.cpu())
        generated_samples = np.array(generated_samples)
        generated_samples = np.concatenate([np.array(i,dtype=float) for i in generated_samples])
    new_generated_data = ad.AnnData(generated_samples)
    
    if retain:
        generated_df = (ori_data[selected_ids].obs).copy()
        ori_df = (ori_data[selected_ids].obs).copy()
        for target_factor_dict in multi_target_list:
            target_factor = target_factor_dict["target_factor"]
            ref_factor_value = str(target_factor_dict["ref_factor_value"])
            tgt_factor_value = str(target_factor_dict["tgt_factor_value"])
            generated_df[target_factor] = generated_df[target_factor].astype(str)
            ori_df[target_factor] = ori_df[target_factor].astype(str)
            if ref_factor_value != "all":
                generated_df[target_factor][np.array(generated_df[target_factor]==ref_factor_value)] = str(tgt_factor_value)
            else:
                generated_df[target_factor][np.array(generated_df[target_factor]!=tgt_factor_value)] = str(tgt_factor_value)
        merged_df = pd.concat([generated_df, ori_df])
        new_generated_data.obs = merged_df
        new_generated_data.obs['Type'] = ['Generated'] * int(len(new_generated_data)/2) + ['Original'] * int(len(new_generated_data)/2)
    else:
        new_generated_data.obs = ori_data.obs

    new_generated_data.write(f"{model_save_pwd}/generated_data_{name}.h5ad")
    return new_generated_data

def factor_value_pool(train_concept_embs, factor_list, ori_data_pwd):
    ori_data = sc.read_h5ad(ori_data_pwd)
    unexplained_factor_pool = train_concept_embs[-1] 
    factor_dict = {}
    for factor in factor_list:
        concept_embs = train_concept_embs[factor_list.index(factor)]
        factor_values = np.unique(ori_data.obs[factor])
        factor_value_dict = {}
        for factor_value in factor_values:
            factor_value_dict[factor_value] = concept_embs[ori_data.obs[factor]==factor_value]
        factor_dict[factor] = factor_value_dict
    factor_dict['unexplained_variables'] = unexplained_factor_pool
    return factor_dict

def concept_embs_sampling_based_factor_dict(ori_data_pwd, factor_dict, factor_list):
    ori_data = sc.read_h5ad(ori_data_pwd)
    new_concept_embs = []
    for factor in factor_list:
        tmp_concept_embs = []
        for factor_value in ori_data.obs[factor]:
            tmp_concept_embs.append(factor_dict[factor][factor_value][np.random.choice(factor_dict[factor][factor_value].shape[0])])
        new_concept_embs.append(np.array(tmp_concept_embs))
    new_concept_embs.append(factor_dict['unexplained_variables'])
    new_concept_embs = np.array(new_concept_embs)
    return new_concept_embs

def target_embs_generation(factor_dict, factor_list, multi_target_list, ori_data_pwd, ref_concept_embs):
    ori_data = sc.read_h5ad(ori_data_pwd)
    for target_factor_dict in multi_target_list:
        target_factor = target_factor_dict["target_factor"]
        tgt_factor_value = target_factor_dict["tgt_factor_value"]
        target_embeddings = factor_dict[target_factor][tgt_factor_value]
        sampled_indices = np.random.choice(target_embeddings.shape[0], size=ref_concept_embs[factor_list.index(target_factor)].shape[0], replace=True)
        generated_tcell_embs = target_embeddings[sampled_indices]
        ref_concept_embs[factor_list.index(target_factor)]=generated_tcell_embs
    return ref_concept_embs

def generation_based_concept_embs(model, concept_embs, save_pwd, 
                                 factor_list, factor_counts, factor_cdag, 
                                 factor_df, name, batch_size):
    train_concept_embs = np.array(concept_embs)
    with torch.no_grad():
        generated_samples = []
        batch_sz = batch_size
        steps = train_concept_embs[0].shape[0]//batch_size + 1
        for ii in range(steps):
            batch_concept_embs = train_concept_embs[:, ii*batch_sz:(ii+1)*batch_sz,:]
            batch_concept_embs = np.transpose(batch_concept_embs, (1,0,2))
            samples_with_cross_attention = model.sample_with_factor(concept_embs = torch.tensor(batch_concept_embs).cuda(), batch_size = batch_concept_embs.shape[0])
            generated_samples.append(samples_with_cross_attention.cpu())
        generated_samples = np.concatenate([np.array(i,dtype=float) for i in generated_samples])
    new_generated_data = ad.AnnData(generated_samples)
    new_generated_data.obs = factor_df
    new_generated_data.write(f"{save_pwd}/{name}.h5ad")
    return new_generated_data

def Simulated_RCT(factor_dict, factor_list, target_factor, number_of_each_factor=500):
    all_randomized_factor_value_embs = []
    target_factor_value_dict = factor_dict[target_factor]
    target_factor_values = np.array(list(target_factor_value_dict.keys()))
    num_of_target_values = len(target_factor_values)
    obs_df = {}
    for factor in factor_list:
        if factor == target_factor:
            factor_value_dict = factor_dict[factor]
            factor_values = np.array(list(factor_value_dict.keys()))
            selected_factor_value_embs = []
            factor_names = []
            for selected_factor_value in factor_values:
                tmp_factor_value_embs = factor_value_dict[selected_factor_value]
                selected_factor_value_embs.append(tmp_factor_value_embs[np.random.choice(len(tmp_factor_value_embs), number_of_each_factor, replace=True)])
                factor_names += [selected_factor_value] * number_of_each_factor
            selected_factor_value_embs = np.concatenate(selected_factor_value_embs, axis=0)
            all_randomized_factor_value_embs.append(selected_factor_value_embs)
            obs_df[factor] = np.array(factor_names).astype(str)
        else:
            factor_value_dict = factor_dict[factor]
            factor_values = np.array(list(factor_value_dict.keys()))
            sampled_indices = np.random.choice(len(factor_values), size=number_of_each_factor, replace=True)
            selected_factor_values = factor_values[sampled_indices]
            selected_factor_value_embs = []
            factor_names = list(selected_factor_values) * num_of_target_values
            obs_df[factor] = np.array(factor_names).astype(str)
            for selected_factor_value in selected_factor_values:
                tmp_factor_value_embs = factor_value_dict[selected_factor_value]
                selected_factor_value_embs.append(tmp_factor_value_embs[np.random.choice(len(tmp_factor_value_embs), 1, replace=True)][0])
            selected_factor_value_embs = np.array(selected_factor_value_embs)
            selected_factor_value_embs = np.concatenate([selected_factor_value_embs] * num_of_target_values, axis=0)
            all_randomized_factor_value_embs.append(selected_factor_value_embs)
    
    selected_factor_value_embs = factor_dict["unexplained_variables"][np.random.choice(len(factor_dict["unexplained_variables"]), number_of_each_factor * num_of_target_values, replace=True)]
    all_randomized_factor_value_embs.append(selected_factor_value_embs)
    all_randomized_factor_value_embs = np.array(all_randomized_factor_value_embs)
    return all_randomized_factor_value_embs, pd.DataFrame(obs_df)