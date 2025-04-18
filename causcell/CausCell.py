import torch
import numpy as np
import pandas as pd
import anndata as ad
import random
from .Modules import Denoise_net, GaussianDiffusion, Trainer
from .Dataset import Dataset
import scanpy as sc
from torch.utils import data
import joblib
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
from copy import deepcopy
from .Metrics_calculation import cal_acc_precision_recall_f1_factors
from pathlib import Path
from .Generation_utils import extract_concept_embs, factor_value_pool, causality_based_concept_embs, target_embs_generation, generation_based_concept_embs


class CausCell:
    """
    CausCell base model class
    """
    def __init__(self, 
                 device = 'cuda', 
                 ema_decay = 0.995, 
                 gradient_accumulate_every = 2, 
                 fp16 = False, 
                 step_start_ema = 2000, 
                 update_ema_every = 1000, 
                 save_and_sample_every = 10000):

        self.device = device
        self.ema_decay = ema_decay
        self.gradient_accumulate_every = gradient_accumulate_every
        self.fp16 = fp16
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every
        self.save_and_sample_every = save_and_sample_every
        self.model = None

    def data_transformation(self, 
                            data_pwd, 
                            save_pwd, 
                            concept_list,
                            log_norm=False):
        data = sc.read_h5ad(data_pwd)
        if log_norm:
            normed_data = data.X / data.X.sum(axis=1)[:,None] * 10000
            exp_data = np.log(normed_data + 1)
        else:
            exp_data = data.X
        new_obs = []
        label_categories = []
        for idx, factor_name in enumerate(concept_list):
            factor_vals = list(data.obs[factor_name].unique())
            val2idx = dict(zip(factor_vals, range(len(factor_vals))))
            new_obs.append([val2idx[i] for i in data.obs[factor_name]])
            label_categories.append(len(factor_vals))
            joblib.dump(val2idx, f"{save_pwd}/{factor_name}_dict.pkl")
        new_df = pd.DataFrame(list(zip(*new_obs)), columns = concept_list)
        new_data = ad.AnnData(exp_data)
        new_data.obs = new_df
        data_name = data_pwd.split("/")[-1]
        new_data.write(f"{save_pwd}/transformed_{data_name}")
        return new_data
    
    def train(self, 
              training_data_pwd, 
              model_save_pwd, 
              concept_list, 
              concept_counts, 
              concept_cdag, 
              *,
              loss_type="l2",
              training_num_steps=100000, 
              training_batch_size=64, 
              training_lr=2e-5,
              max_profile_size=2000, 
              timesteps=1000, 
              seed = 888, 
              train_log = True):
        # random seed setting
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        
        # read training dataset
        train_data = sc.read_h5ad(training_data_pwd)
        training_output_pwd = model_save_pwd
        if train_data.X.shape[1] > max_profile_size:
            training_profile_size = max_profile_size
        else:
            training_profile_size = train_data.X.shape[1]
        model = Denoise_net(training_profile_size, training_profile_size, len(concept_counts) + 1, torch.tensor(concept_cdag).float(), concept_counts).to(self.device)
        diffusion_model = GaussianDiffusion(model, profile_size = training_profile_size, timesteps = timesteps, loss_type = loss_type).to(self.device)
        trainer = Trainer(diffusion_model, training_data_pwd, concept_list,
                profile_size = training_profile_size, 
                train_batch_size = training_batch_size, 
                train_lr = training_lr,
                train_num_steps = training_num_steps, 
                results_folder = training_output_pwd,
                train_log = train_log, 
                ema_decay = self.ema_decay, 
                gradient_accumulate_every = self.gradient_accumulate_every, 
                fp16 = self.fp16,
                step_start_ema = self.step_start_ema, 
                update_ema_every = self.update_ema_every, 
                save_and_sample_every = self.save_and_sample_every)
        
        trainer.train()
        self.model = deepcopy(diffusion_model)

    def load_trained(self, 
                     concept_list, 
                     concept_counts, 
                     concept_cdag, 
                     results_folder, 
                     *,
                     trained_profile_size=1000, 
                     milestone=10, 
                     timesteps=1000):
        factor_order = concept_list
        profile_size = trained_profile_size
        model_denoise = Denoise_net(profile_size, profile_size, len(concept_counts) + 1, torch.tensor(concept_cdag).float(), concept_counts).cuda()
        GaussianDiffusion_model = GaussianDiffusion(model_denoise, profile_size = profile_size, timesteps = timesteps).cuda()
        data = torch.load(str(Path(results_folder) / f'model-{milestone}.pt'))
        GaussianDiffusion_model.load_state_dict(data['model'])
        self.model = deepcopy(GaussianDiffusion_model)
    
    def sampling_cells(self, 
                       testing_data_pwd, 
                       concept_list, 
                       concept_counts, 
                       concept_cdag, 
                       *,
                       profile_size=1000):
        with torch.no_grad():
            dataset = Dataset(testing_data_pwd, profile_size, concept_list)
            dataloader = data.DataLoader(dataset, batch_size = 1280, shuffle = False, pin_memory = True)
            disentanglement_embs = []
            generated_samples = []
            for idx, data_ in enumerate(dataloader):
                batch_concept_embs = self.model.denosie_fn.DisentanglementEncoder.eval()(data_[0].cuda(), data_[1].cuda())[0].cpu()
                disentanglement_embs.append(batch_concept_embs)
                samples_with_cross_attention = self.model.sample_with_factor(concept_embs = batch_concept_embs.cuda(), batch_size = len(data_[0]))
                generated_samples.append(samples_with_cross_attention.cpu())
            if len(generated_samples) >= 2:
                # generated_samples = np.array(generated_samples)
                generated_samples = np.concatenate([np.array(i,dtype=float) for i in generated_samples])
            else:
                generated_samples = np.array(generated_samples[0], dtype=float)
        return generated_samples
    
    def sampling_concepts(self, 
                          testing_data_pwd, 
                          concept_list,
                          concept_counts, 
                          concept_cdag, 
                          *, 
                          profile_size = 1000):
        with torch.no_grad():
            dataset = Dataset(testing_data_pwd, profile_size, concept_list)
            dataloader = data.DataLoader(dataset, batch_size = 1280, shuffle = False, pin_memory = True)
            disentanglement_embs = []
            for idx, data_ in enumerate(dataloader):
                batch_concept_embs = self.model.denosie_fn.DisentanglementEncoder.eval()(data_[0].cuda(), data_[1].cuda())[0].cpu()
                disentanglement_embs.append(batch_concept_embs)
            
            concept_embs = [[] for _ in range(len(concept_list) + 1)]
            for factor in range(len(concept_list) + 1):
                for idx, i in enumerate(disentanglement_embs):
                    for b in i:
                        concept_embs[factor].append(b[factor])
            
            concept_embs_stacked = []
            for i in concept_embs:
                concept_embs_stacked.append(np.array(torch.stack(i)))
        return np.array(concept_embs_stacked)
    
    def disentanglement(self, 
                        testing_data_pwd, 
                        saved_pwd,
                        concept_list, 
                        concept_counts, 
                        concept_cdag, 
                        *, 
                        sampling_counts = 10,
                        profile_size = 1000):
        concept_embs = self.sampling_concepts(testing_data_pwd, concept_list, concept_counts, concept_cdag, profile_size = profile_size)
        for i in range(sampling_counts - 1):
            concept_embs += self.sampling_concepts(testing_data_pwd, concept_list, concept_counts, concept_cdag, profile_size = profile_size)
        concept_embs /= sampling_counts
        if not os.path.exists(saved_pwd):
            os.mkdir(saved_pwd)
        joblib.dump(concept_embs, saved_pwd + f"/factors_embs.pkl")
        return concept_embs
    
    def concept_prediction(self, testing_data_pwd, concept_embs, concept_list, concept_counts, concept_cdag):
        # multi-label classification
        factor_scores = []
        with torch.no_grad():
            for idx, predictor in enumerate(self.model.denosie_fn.DisentanglementEncoder.label_predictor):
                scores = predictor.eval()(torch.tensor(concept_embs[idx]).cuda())
                factor_scores.append(np.array(scores.cpu()))
        pred_labels = [i.argmax(axis = 1) for i in factor_scores]
        test_data = sc.read_h5ad(testing_data_pwd)
        ground_labels = [np.array(test_data.obs[i]) for i in concept_list]

        accs, precisions, recalls, f1s = cal_acc_precision_recall_f1_factors(pred_labels, ground_labels)
        for idx in range(len(accs)):
            print(f"Factor_{concept_list[idx]}: ACC:{accs[idx]}\tPrecision:{precisions[idx]}\tRecall:{recalls[idx]}\tF1:{f1s[idx]}")
            
    def get_generated_cells(self, 
                            testing_data_pwd, 
                            saved_pwd, 
                            concept_list, 
                            concept_counts, 
                            concept_cdag, 
                            *, 
                            sampling_counts = 10,
                            profile_size = 1000):
        generated_samples = self.sampling_cells(testing_data_pwd, concept_list, concept_counts, concept_cdag, profile_size = profile_size)
        for _ in range(sampling_counts - 1):
            generated_samples += self.sampling_cells(testing_data_pwd, concept_list, concept_counts, concept_cdag, profile_size = profile_size)
        generated_samples /= sampling_counts
        if not os.path.exists(saved_pwd):
            os.mkdir(saved_pwd)
        joblib.dump(generated_samples, saved_pwd + f"/generated_cells.pkl")
        return generated_samples
    
    def counterfactual_generation(self, 
                                  data_pwd, 
                                  save_pwd, 
                                  concept_list, 
                                  concept_counts, 
                                  concept_cdag, 
                                  multi_target_list, 
                                  file_name, 
                                  *,
                                  batch_size = 32):
        concept_embs = extract_concept_embs(data_pwd, self.model, 
                                            self.model.denosie_fn.DisentanglementEncoder.exogenous_encoder_m_v[0].weight.size()[1], 
                                            concept_list, concept_counts, concept_cdag, batch_size)
        factor_dict = factor_value_pool(concept_embs, concept_list, data_pwd)
        ori_data = sc.read_h5ad(data_pwd)
        selected_ids = np.array([True]*len(concept_embs[0]))
        for target_factor_dict in multi_target_list:
            target_factor = target_factor_dict["target_factor"]
            ref_factor_value = target_factor_dict["ref_factor_value"]
            tgt_factor_value = target_factor_dict["tgt_factor_value"]
            selected_ids = selected_ids & np.array((ori_data.obs[target_factor]==ref_factor_value))
        ref_concept_embs = concept_embs[:,selected_ids,:].copy()
        final_embs_ori = causality_based_concept_embs(ref_concept_embs, concept_cdag)
        
        target_embs = target_embs_generation(factor_dict, concept_list, multi_target_list, data_pwd, ref_concept_embs)
        final_embs_target = causality_based_concept_embs(target_embs, concept_cdag)
        new_df = ori_data[selected_ids].obs.copy()
        ori_df = ori_data[selected_ids].obs.copy()
        for target_factor_dict in multi_target_list:
            target_factor = target_factor_dict["target_factor"]
            tgt_factor_value = target_factor_dict["tgt_factor_value"]
            new_df[target_factor] = [str(tgt_factor_value)] * len(new_df[target_factor])
            ori_df[target_factor] = ori_df[target_factor].astype(str)

        final_embs = np.concatenate([final_embs_target, final_embs_ori], axis = 1)
        final_df = pd.concat([new_df, ori_df])
        final_df['Type'] = ['Generated'] * len(new_df) + ['Original'] * len(ori_df)
        new_generated_data = generation_based_concept_embs(self.model, final_embs, save_pwd, 
                                                           concept_list, concept_counts, concept_cdag, 
                                                           final_df, file_name, batch_size)
        return new_generated_data