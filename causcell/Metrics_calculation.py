from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, KMeans, SpectralClustering, DBSCAN
import scanpy as sc
import anndata as ad
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

np.random.seed(888)

# calculate the pearson correlation between real samples and generated samples
# this is the mean PCC based all samples
def cal_pearson_correlation(real_samples, generated_samples):
    pccs = [
        pearsonr(real_samples[i], generated_samples[i])[0] for i in range(len(real_samples))
    ]
    
    return np.array(pccs).mean()

# calculate the direct PCC between real samples and generated samples and PCC with random mismatch as control
def cal_pearson_correlation_with_CT(real_samples, generated_samples, seed = 888):
    trial_PCC = cal_pearson_correlation(real_samples, generated_samples)
    np.random.seed(seed)
    shuffle_idx = np.array(range(len(real_samples)))
    np.random.shuffle(shuffle_idx)
    CT_PCC = cal_pearson_correlation(real_samples, generated_samples[shuffle_idx])
    
    return trial_PCC, CT_PCC

def cal_MSE(real_samples, generated_samples):
    mse = ((real_samples - generated_samples)**2).mean()
    
    return mse

def cal_MSE_CT(real_samples, generated_samples, seed = 888):
    trial_MSE = cal_MSE(real_samples, generated_samples)
    np.random.seed(seed)
    shuffle_idx = np.array(range(len(real_samples)))
    np.random.shuffle(shuffle_idx)
    CT_MSE = cal_MSE(real_samples, generated_samples[shuffle_idx])

    return trial_MSE, CT_MSE

def calculate_ARI_NMI(cluster_list1, cluster_list2):
    """
    Calculate the ARI and NMI metric given two cluster list
    
    Parameters:
    - cluster_list1: np.array or list, cluster idx list
    - cluster_list2: np.array or list, cluster idx list
    
    Return:
    - float, float: ARI score and NMI score
    """
    
    # calculated ARI scores
    ARI_score = adjusted_rand_score(cluster_list1, cluster_list2)
    # calculated NMI scores
    nmi_score = normalized_mutual_info_score(cluster_list1, cluster_list2)
    return ARI_score, nmi_score

def embs_based_clustering_methods(
    data, 
    n_clusters, 
    cluster_method = 'Hierarchical', 
    ):
    """
    Embedding based clustering methods;
    Please note that the clustering number is invalid for AffinityPropagation and DBSCAN methods
    
    Parameters:
    - data: np.array, the dataset
    - n_clusters: int, the number of clusters used for clustering
    - cluster_method: str, the method used for clustering, 'Hierarchical' is default
    
    Return:
    - np.array: the cluster label for each sample
    """
    if cluster_method == 'Hierarchical':
        clustering = AgglomerativeClustering(n_clusters).fit(data)
        return clustering.labels_
    elif cluster_method == 'AffinityPropagation':
        clustering = AffinityPropagation(random_state=5).fit(data)
        return clustering.labels_
    elif cluster_method == 'Kmeans':
        clustering = KMeans(n_clusters, random_state=5, n_init="auto").fit(data)
        return clustering.labels_
    elif cluster_method == 'SpectralClustering':
        clustering = SpectralClustering(n_clusters, random_state=5, assign_labels="discretize").fit(data)
        return clustering.labels_
    elif cluster_method == 'DBSCAN':
        clustering = DBSCAN().fit(data)
        return clustering.labels_
    else:
        print("Please select the clustering method from the implemented method lists:")
        print("AffinityPropagation, Hierarchical, KMeans, SpectralClustering, DBSCAN")
        print("Other clustering methods will be incorporated in the future")

def cal_ARI_NMI(real_samples, generated_samples, n_clusters = 8):
    labels_real = embs_based_clustering_methods(real_samples, n_clusters)
    labels_generated = embs_based_clustering_methods(generated_samples, n_clusters)
    ARI_score, NMI_score = calculate_ARI_NMI(labels_real, labels_generated)
    
    return ARI_score, NMI_score, labels_real, labels_generated

def marker_gene_matching_score(real_samples, generated_samples, obs_df, top_rank = 50):
    real_adata = ad.AnnData(X = real_samples)
    generated_adata = ad.AnnData(X = generated_samples)
    tuple_list = []
    for idx in range(len(obs_df)):
        tuple_list.append(str([obs_df[i][idx] for i in obs_df]))
    unique_tuple_list = np.unique(tuple_list)
    tmp_dict = dict(zip(unique_tuple_list, range(len(unique_tuple_list))))
    merged_class = np.array([tmp_dict[i] for i in tuple_list])

    
    unique_elements, element_freq = np.unique(merged_class, return_counts=True)
    rm_unique_elements = unique_elements[element_freq==1]
    if len(rm_unique_elements) > 0:
        retained_idxs = np.array([True] * len(merged_class))
        for rue in rm_unique_elements:
            retained_idxs[merged_class == rue]=False
        merged_class = merged_class[retained_idxs]
        real_adata = ad.AnnData(X = real_samples[retained_idxs])
        generated_adata = ad.AnnData(X = generated_samples[retained_idxs])
    
    real_adata.obs['Group'] = merged_class
    generated_adata.obs['Group'] = merged_class    
        
    real_adata.obs['Group'] = real_adata.obs['Group'].astype('category')
    generated_adata.obs['Group'] = generated_adata.obs['Group'].astype('category')
    
    sc.tl.rank_genes_groups(real_adata, 'Group', method = 'wilcoxon')
    sc.tl.rank_genes_groups(generated_adata, 'Group', method = 'wilcoxon')
    
    count_intersections = 0
    for idx, i in enumerate(merged_class):
        top_rank_genes_real = real_adata.uns['rank_genes_groups']['names'][str(i)][:top_rank]
        top_rank_genes_generated = generated_adata.uns['rank_genes_groups']['names'][str(i)][:top_rank]
        intersections = np.intersect1d(top_rank_genes_real, top_rank_genes_generated)
        count_intersections += len(intersections)
    score = count_intersections / (len(merged_class) * top_rank)
    return score

def cal_marker_gene_matching_score_CT(real_samples, generated_samples, obs_df, top_rank = 50, seed = 888):
    score = marker_gene_matching_score(real_samples, generated_samples, obs_df, top_rank = top_rank)
    np.random.seed(seed)
    shuffle_idx = np.array(range(len(real_samples)))
    np.random.shuffle(shuffle_idx)
    score_CT = marker_gene_matching_score(real_samples, generated_samples[shuffle_idx], obs_df, top_rank = top_rank)
    return score, score_CT

def cal_acc_precision_recall_f1(pred_labels, ground_labels, weighted = True):
    acc = accuracy_score(ground_labels, pred_labels)
    if weighted:
        precision = precision_score(ground_labels, pred_labels, average='weighted')
        recall = recall_score(ground_labels, pred_labels, average='weighted')
        f1 = f1_score(ground_labels, pred_labels, average='weighted')     
    else:
        precision = precision_score(ground_labels, pred_labels)
        recall = recall_score(ground_labels, pred_labels)
        f1 = f1_score(ground_labels, pred_labels)
    return acc, precision, recall, f1

def cal_acc_precision_recall_f1_factors(pred_labels_list, ground_labels_list):
    accs = []
    precisions = []
    recalls = []
    f1s = []
    for idx in range(len(pred_labels_list)):
        acc, precision, recall, f1 = cal_acc_precision_recall_f1(pred_labels_list[idx], ground_labels_list[idx])
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return accs, precisions, recalls, f1s