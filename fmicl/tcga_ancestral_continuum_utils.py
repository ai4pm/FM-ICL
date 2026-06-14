import os
import numpy as np 
import pandas as pd 
import numpy as np
from collections import deque

from helper_code.preProcess import get_Methylation, get_MicroRNA, get_mRNA

# ====================================================
# ------------ modedules for creating datasets for 
# the tasks along the ancestral continuum  -----------
# ====================================================
# loading data 
def create_ancestral_continuum_dataset(train_data, test_data, expression_data, radius, min_num=5, min_ratio=0.1):
    """
    For each sample in test_data, find training samples (from train_data)
    whose genetic features fall within a given radius, and then collect
    their expression features as the training set for that test sample.

    Before doing this, samples in train_data and test_data that do NOT
    exist in expression_data are removed.

    Parameters
    ----------
    train_data : dict[str, np.ndarray]
        Mapping from training sample ID -> genetic feature vector (1D array).
    test_data : dict[str, np.ndarray]
        Mapping from test sample ID -> genetic feature vector (1D array).
    expression_data : dict[str, np.ndarray]
        Mapping from sample ID -> expression feature vector (1D array).
        Must contain entries for all retained training and test sample IDs.
    radius : float
        Radius in genetic feature space for neighbor selection.

    Returns
    -------
    list[dict]
        Mapping from test sample ID -> 2D array of expression features
        of training neighbors within radius.
        Shape is (n_neighbors_for_this_test, expr_dim).
        If no neighbors are found, an empty array of shape (0, expr_dim)
        is returned for that test sample.
    """
    # Filter train_data and test_data to only samples that have expression data 
    expr_ids = set(expression_data['Samples'])

    filtered_train_data = {
        sid: vec for sid, vec in train_data.items() if sid in expr_ids
    }
    filtered_test_data = {
        sid: vec for sid, vec in test_data.items() if sid in expr_ids
    }

    sample_ind = {} # a hashmap for sample names and their indices in the expression data
    for i in range(len(expression_data['Samples'])):
        sample_name = expression_data['Samples'][i]
        sample_ind[sample_name] = i 


    # Prepare training genetic matrix
    train_ids = list(filtered_train_data.keys())
    train_genetic = np.stack([filtered_train_data[sid] for sid in train_ids], axis=0)

    result = {}
    valid_test_id = []

    # An array that saves the entire population excluding test set and the local training set
    expression_exclude_local_training_set = []
    all_expression = []
    expression_train_ind = []

    # For each test sample, find neighbors in genetic space and collect expression 
    for test_id, test_vec in filtered_test_data.items():
        test_vec = np.asarray(test_vec)

        # Distances from this test sample to all training samples
        diff = train_genetic[:, :2] - test_vec[:2]  # shape: (n_train, genetic_dim)
        dists = np.linalg.norm(diff, axis=1)  # shape: (n_train,)

        # Indices of training samples within the radius
        genetic_neighbor_ind = np.where(dists <= radius)[0]
        genetic_other_ind = np.array(list(set(np.arange(len(train_genetic))) - set(list(genetic_neighbor_ind))))
        expression_neighbor_ind = []
        expression_other_ind = []
        for idx in genetic_neighbor_ind:
            expression_neighbor_ind.append(sample_ind[train_ids[idx]])
        for idx in genetic_other_ind:
            expression_other_ind.append(sample_ind[train_ids[idx]])
        other_inx = np.array(expression_other_ind)
        expression_neighbor_ind = np.array(expression_neighbor_ind)
        if len(expression_neighbor_ind) > 0:
            feat = expression_data['X'][expression_neighbor_ind]
            lab = expression_data['Y'][expression_neighbor_ind]
            valid_test_id.append(test_id)
        else:
            continue 

        # check if the data fits for training, e.g., label ratio and training size
        unique, counts = np.unique(lab, return_counts=True)
        try:
            ratio = counts.min() / counts.sum()
        except:
            continue
        if len(counts) < 2 or min(counts) < min_num or ratio < min_ratio:
            continue
        result[test_id] = (feat, lab)
        # print('test id:', test_id, ' | neighbor size:', len(feat), ' | label ratio:', ratio)
        expression_exclude_local_training_set.append((expression_data['X'][other_inx], expression_data['Y'][other_inx]))
    # get all training and test data 
    test_ind = []
    for sid in valid_test_id:
        test_ind.append(sample_ind[sid])
    
    train_sample_names = []
    for sid in filtered_train_data:
        expression_train_ind.append(sample_ind[sid])
        train_sample_names.append(sid)
    expression_train_ind = np.array(expression_train_ind)
    all_test_expression = (expression_data['X'][test_ind], expression_data['Y'][test_ind])
    all_expression = (expression_data['X'][expression_train_ind], expression_data['Y'][expression_train_ind])
    return train_sample_names, all_expression, expression_exclude_local_training_set, all_test_expression, result

def create_ancestral_continuum_dataset_regression(train_data, test_data, expression_data, radius, min_num=5):
    """
    For each sample in test_data, find training samples (from train_data)
    whose genetic features fall within a given radius, and then collect
    their expression features as the training set for that test sample.

    Before doing this, samples in train_data and test_data that do NOT
    exist in expression_data are removed.

    Parameters
    ----------
    train_data : dict[str, np.ndarray]
        Mapping from training sample ID -> genetic feature vector (1D array).
    test_data : dict[str, np.ndarray]
        Mapping from test sample ID -> genetic feature vector (1D array).
    expression_data : dict[str, np.ndarray]
        Mapping from sample ID -> expression feature vector (1D array).
        Must contain entries for all retained training and test sample IDs.
    radius : float
        Radius in genetic feature space for neighbor selection.

    Returns
    -------
    list[dict]
        Mapping from test sample ID -> 2D array of expression features
        of training neighbors within radius.
        Shape is (n_neighbors_for_this_test, expr_dim).
        If no neighbors are found, an empty array of shape (0, expr_dim)
        is returned for that test sample.
    """
    # Filter train_data and test_data to only samples that have expression data 
    expr_ids = set(expression_data['Samples'])

    filtered_train_data = {
        sid: vec for sid, vec in train_data.items() if sid in expr_ids
    }
    filtered_test_data = {
        sid: vec for sid, vec in test_data.items() if sid in expr_ids
    }

    sample_ind = {} # a hashmap for sample names and their indices in the expression data
    for i in range(len(expression_data['Samples'])):
        sample_name = expression_data['Samples'][i]
        sample_ind[sample_name] = i 


    # Prepare training genetic matrix
    train_ids = list(filtered_train_data.keys())
    train_genetic = np.stack([filtered_train_data[sid] for sid in train_ids], axis=0)

    result = {}
    valid_test_id = []

    # An array that saves the entire population excluding test set and the local training set
    expression_exclude_local_training_set = []
    all_expression = []
    expression_train_ind = []

    # For each test sample, find neighbors in genetic space and collect expression 
    for test_id, test_vec in filtered_test_data.items():
        test_vec = np.asarray(test_vec)

        # Distances from this test sample to all training samples
        diff = train_genetic[:, :2] - test_vec[:2]  # shape: (n_train, genetic_dim)
        dists = np.linalg.norm(diff, axis=1)  # shape: (n_train,)

        # Indices of training samples within the radius
        genetic_neighbor_ind = np.where(dists <= radius)[0]
        genetic_other_ind = np.array(list(set(np.arange(len(train_genetic))) - set(list(genetic_neighbor_ind))))
        expression_neighbor_ind = []
        expression_other_ind = []
        for idx in genetic_neighbor_ind:
            expression_neighbor_ind.append(sample_ind[train_ids[idx]])
        for idx in genetic_other_ind:
            expression_other_ind.append(sample_ind[train_ids[idx]])
        other_inx = np.array(expression_other_ind)
        expression_neighbor_ind = np.array(expression_neighbor_ind)
        if len(expression_neighbor_ind) > 1:
            feat = expression_data['X'][expression_neighbor_ind]
            lab = expression_data['Y'][expression_neighbor_ind]
            censor = expression_data['C'][expression_neighbor_ind]
            valid_test_id.append(test_id)
        else:
            continue 


        result[test_id] = (feat, lab, censor)
        print('test id:', test_id, ' | neighbor size:', len(feat))
        expression_exclude_local_training_set.append((expression_data['X'][other_inx], expression_data['Y'][other_inx], expression_data['C'][other_inx]))
    # get all training and test data 
    test_ind = []
    for sid in valid_test_id:
        test_ind.append(sample_ind[sid])
    train_sample_names = []
    for sid in filtered_train_data:
        expression_train_ind.append(sample_ind[sid])
        train_sample_names.append(sid)
    expression_train_ind = np.array(expression_train_ind)
    all_test_expression = (expression_data['X'][test_ind],  expression_data['Y'][test_ind], expression_data['C'][test_ind])
    all_expression = (expression_data['X'][expression_train_ind], expression_data['Y'][expression_train_ind], expression_data['C'][expression_train_ind])
    return all_expression, expression_exclude_local_training_set, all_test_expression, result


def create_ancestral_continuum_dataset_hybrid(train_data, 
                                              test_data, 
                                              expression_data, 
                                              selected_id = None,
                                              k_neighbors=20, 
                                              radius=20, 
                                              min_num=5,
                                              min_ratio=0.1):
    """
    For each sample in test_data, find training samples (from train_data)
    whose genetic features fall within a given radius, and then collect
    their expression features as the training set for that test sample.

    Before doing this, samples in train_data and test_data that do NOT
    exist in expression_data are removed.

    Parameters
    ----------
    train_data : dict[str, np.ndarray]
        Mapping from training sample ID -> genetic feature vector (1D array).
    test_data : dict[str, np.ndarray]
        Mapping from test sample ID -> genetic feature vector (1D array).
    expression_data : dict[str, np.ndarray]
        Mapping from sample ID -> expression feature vector (1D array).
        Must contain entries for all retained training and test sample IDs.
    radius : float
        Radius in genetic feature space for neighbor selection.

    Returns
    -------
    list[dict]
        Mapping from test sample ID -> 2D array of expression features
        of training neighbors within radius.
        Shape is (n_neighbors_for_this_test, expr_dim).
        If no neighbors are found, an empty array of shape (0, expr_dim)
        is returned for that test sample.
    """
    # Filter train_data and test_data to only samples that have expression data
    expr_ids = set(expression_data['Samples'])
    selected_set = set(selected_id) if selected_id is not None else None
    filtered_train_data = {sid: vec for sid, vec in train_data.items() if sid in expr_ids}
    filtered_test_data  = {sid: vec for sid, vec in test_data.items()  if sid in expr_ids and (selected_set is None or sid in selected_set)}


    sample_ind = {} # a hashmap for sample names and their indices in the expression data
    for i in range(len(expression_data['Samples'])):
        sample_name = expression_data['Samples'][i]
        sample_ind[sample_name] = i 


    # Prepare training genetic matrix
    train_ids = list(filtered_train_data.keys())
    train_genetic = np.stack([filtered_train_data[sid] for sid in train_ids], axis=0)

    result = {}
    valid_test_id = []

    # An array that saves the entire population excluding test set and the local training set
    expression_exclude_local_training_set = []
    all_expression = []
    expression_train_ind = []

    # For each test sample, find neighbors in genetic space and collect expression 
    for test_id, test_vec in filtered_test_data.items():
        test_vec = np.asarray(test_vec)

        # Distances from this test sample to all training samples
        diff = train_genetic[:, :2] - test_vec[:2]  # shape: (n_train, genetic_dim)
        dists = np.linalg.norm(diff, axis=1)  # shape: (n_train,)

        # Indices of k nearest training samples
        k = min(k_neighbors, len(train_genetic))
        # Indices of training samples within the radius
        genetic_neighbor_ind = np.where(dists <= radius)[0]
        if len(genetic_neighbor_ind) < k:
            genetic_neighbor_ind = np.argsort(dists)[:k]
        
        genetic_neighbor_ind = np.array(genetic_neighbor_ind)

        genetic_other_ind = np.array(list(set(np.arange(len(train_genetic))) - set(list(genetic_neighbor_ind))))
        expression_neighbor_ind = []
        expression_other_ind = []
        for idx in genetic_neighbor_ind:
            expression_neighbor_ind.append(sample_ind[train_ids[idx]])
        for idx in genetic_other_ind:
            expression_other_ind.append(sample_ind[train_ids[idx]])
        other_inx = np.array(expression_other_ind)
        expression_neighbor_ind = np.array(expression_neighbor_ind)
        if len(expression_neighbor_ind) > 0:
            feat = expression_data['X'][expression_neighbor_ind]
            lab = expression_data['Y'][expression_neighbor_ind]
        else:
            continue 

        # check if the data fits for training, e.g., label ratio and training size
        unique, counts = np.unique(lab, return_counts=True)
        if counts.sum() == 0:
            continue
        ratio = counts.min() / counts.sum()
        if len(counts) < 2 or min(counts) < min_num or ratio < min_ratio or len(unique) == 1:
            continue
        valid_test_id.append(test_id)
        result[test_id] = (feat, lab)
        # print('test id:', test_id, ' | neighbor size:', len(feat), ' | label ratio:', ratio)
        expression_exclude_local_training_set.append((expression_data['X'][other_inx], expression_data['Y'][other_inx]))
    # get all training and test data 
    test_ind = []
    if len(valid_test_id) == 0:
        raise ValueError("No valid test samples found after neighbor selection. Consider adjusting radius or min_num.")
    for sid in valid_test_id:
        test_ind.append(sample_ind[sid])
    test_ind = np.array(test_ind, dtype=int)
    
    train_sample_names = []
    for sid in filtered_train_data:
        expression_train_ind.append(sample_ind[sid])
        train_sample_names.append(sid)
    expression_train_ind = np.array(expression_train_ind)
    all_test_expression = (expression_data['X'][test_ind], expression_data['Y'][test_ind])
    all_expression = (expression_data['X'][expression_train_ind], expression_data['Y'][expression_train_ind])
    return train_sample_names, all_expression, expression_exclude_local_training_set, all_test_expression, result

def create_ancestral_continuum_dataset_dbscan(
    train_data,
    test_data,
    expression_data,
    selected_id = None, 
    radius=20,        # used as DBSCAN eps
    min_num=5,
    min_ratio=0.1
):
    """
    For each sample in test_data, find training samples whose genetic features are
    density-reachable from the test point under a DBSCAN-like rule:
      - eps = radius
      - min_pts = k_neighbors
    Then collect their expression features as the training set for that test sample.

    Notes:
    - Expansion occurs only through core points (DBSCAN rule).
    - If the seed neighborhood of the test point contains no core points,
      we return the seed points only (no expansion).
    """

    # Filter train_data and test_data to only samples that have expression data
    expr_ids = set(expression_data['Samples'])
    selected_set = set(selected_id) if selected_id is not None else None
    filtered_train_data = {sid: vec for sid, vec in train_data.items() if sid in expr_ids}
    filtered_test_data  = {sid: vec for sid, vec in test_data.items()  if sid in expr_ids and (selected_set is None or sid in selected_set)}

    # sample name -> index in expression_data
    sample_ind = {name: i for i, name in enumerate(expression_data['Samples'])}

    # Prepare training genetic matrix
    train_ids = list(filtered_train_data.keys())
    train_genetic = np.stack([filtered_train_data[sid] for sid in train_ids], axis=0)

    result = {}
    valid_test_id = []
    expression_exclude_local_training_set = []
    expression_train_ind = []

    # ---- DBSCAN-style precompute on training set (using first 2 genetic dims, same as your code) ----
    X2 = train_genetic[:, :2].astype(float)
    n_train = X2.shape[0]
    if n_train == 0:
        # Nothing to do
        return [], (np.empty((0,)), np.empty((0,))), [], (np.empty((0,)), np.empty((0,))), {}

    # Pairwise distances (O(n^2)). If n_train is huge, tell me and I’ll swap in a KDTree/ANN version.
    diff = X2[:, None, :] - X2[None, :, :]
    dist_mat = np.linalg.norm(diff, axis=2)

    # neighbors[i] = indices within eps of i (includes itself)
    neighbors = [np.where(dist_mat[i] <= radius)[0] for i in range(n_train)]

    # core point mask under min_pts 
    min_pts = max(1, int(min_num))
    core_mask = np.array([len(neighbors[i]) >= min_pts for i in range(n_train)], dtype=bool)

    def density_reachable_from_seed(seed_idx: np.ndarray) -> np.ndarray:
        """
        Expand from a seed set using DBSCAN density-reachability:
        - Add neighbors of core points; border points do not expand.
        - If there are no core points in seed, return seed only.
        """
        seed_idx = np.unique(seed_idx).astype(int)
        if seed_idx.size == 0:
            return seed_idx

        # If seed touches no core points, DBSCAN expansion can't proceed
        if not np.any(core_mask[seed_idx]):
            return seed_idx

        reachable = set(map(int, seed_idx))
        q = deque([int(i) for i in seed_idx])

        while q:
            cur = q.popleft()
            if not core_mask[cur]:
                continue  # expand only from core points

            for nb in neighbors[cur]:
                nb = int(nb)
                if nb not in reachable:
                    reachable.add(nb)
                    q.append(nb)

        return np.array(sorted(reachable), dtype=int)

    # ---- Main loop over test points ----
    for test_id, test_vec in filtered_test_data.items():
        test_vec = np.asarray(test_vec, dtype=float)
        p2 = test_vec[:2]

        # Seed = training points within eps of p
        dists_to_p = np.linalg.norm(X2 - p2[None, :], axis=1)
        seed = np.where(dists_to_p <= radius)[0]

        # DBSCAN-style reachable region from seed
        genetic_neighbor_ind = density_reachable_from_seed(seed)

        # (Optional safeguard) If nothing found, skip
        if genetic_neighbor_ind.size == 0:
            continue

        # Build "other" set (all remaining training points)
        genetic_neighbor_set = set(map(int, genetic_neighbor_ind))
        genetic_other_ind = np.array([i for i in range(n_train) if i not in genetic_neighbor_set], dtype=int)

        # Map genetic indices -> expression indices
        expression_neighbor_ind = np.array([sample_ind[train_ids[idx]] for idx in genetic_neighbor_ind], dtype=int)
        expression_other_ind    = np.array([sample_ind[train_ids[idx]] for idx in genetic_other_ind], dtype=int)

        feat = expression_data['X'][expression_neighbor_ind]
        lab  = expression_data['Y'][expression_neighbor_ind]

        # check if the data fits for training (label ratio and training size)
        unique, counts = np.unique(lab, return_counts=True)
        if counts.sum() == 0:
            continue
        ratio = counts.min() / counts.sum()
        if len(counts) < 2 or min(counts) < min_num or ratio < min_ratio:
            continue

        valid_test_id.append(test_id)
        result[test_id] = (feat, lab)
        # print('test id:', test_id, ' | neighbor size:', len(feat), ' | label ratio:', ratio)
        expression_exclude_local_training_set.append(
            (expression_data['X'][expression_other_ind], expression_data['Y'][expression_other_ind])
        )

    # ---- Aggregate outputs (same as your original) ----
    test_ind = np.array([sample_ind[sid] for sid in valid_test_id], dtype=int)

    train_sample_names = []
    for sid in filtered_train_data:
        expression_train_ind.append(sample_ind[sid])
        train_sample_names.append(sid)
    expression_train_ind = np.array(expression_train_ind, dtype=int)

    all_test_expression = (expression_data['X'][test_ind], expression_data['Y'][test_ind])
    all_expression      = (expression_data['X'][expression_train_ind], expression_data['Y'][expression_train_ind])

    return train_sample_names, all_expression, expression_exclude_local_training_set, all_test_expression, result

def create_ancestral_continuum_dataset_pmao(
                        train_data,
                        test_data,
                        expression_data,
                        selected_id,
                        ancestry_dim=2,
                        ):
    # Filter train_data and test_data to only samples that have expression data
    expr_ids = set(expression_data['Samples'])
    selected_set = set(selected_id) if selected_id is not None else None
    filtered_train_data = {sid: vec for sid, vec in train_data.items() if sid in expr_ids}
    filtered_test_data  = {sid: vec for sid, vec in test_data.items()  if sid in expr_ids and (selected_set is None or sid in selected_set)}

    if len(filtered_test_data) == 0:
        raise ValueError("No valid test samples found after filtering.")

    sample_ind = {} # a hashmap for sample names and their indices in the expression data
    for i in range(len(expression_data['Samples'])):
        sample_name = expression_data['Samples'][i]
        sample_ind[sample_name] = i 


    test_ancestry_omics = {}
    train_ancestry_omics = {}
    valid_test_id = []
    valid_train_id = []

    # For each test sample, find neighbors in genetic space and collect expression 
    for test_id, test_vec in filtered_test_data.items():
        test_genetic_feat = np.asarray(test_vec)[:ancestry_dim].reshape((1, -1)) # select a couple of pcs at the beginning
        test_omics_feat = expression_data['X'][sample_ind[test_id]].reshape((1, -1))
        test_lab = expression_data['Y'][sample_ind[test_id]]
        test_feat = np.hstack((test_omics_feat, test_genetic_feat))

        valid_test_id.append(test_id)
        test_ancestry_omics[test_id] = (test_feat, test_lab)

    for train_id, train_vec in filtered_train_data.items():
        train_genetic_feat = np.asarray(train_vec)[:ancestry_dim].reshape((1, -1)) # select a couple of pcs at the beginning
        train_omics_feat = expression_data['X'][sample_ind[train_id]].reshape((1, -1))
        train_lab = expression_data['Y'][sample_ind[train_id]]
        train_feat = np.hstack((train_omics_feat, train_genetic_feat))

        valid_train_id.append(train_id)
        train_ancestry_omics[train_id] = (train_feat, train_lab)

    return (valid_train_id,
            valid_test_id,
            train_ancestry_omics,
            test_ancestry_omics)

# The following version saves genetic and omics in various places 
def create_ancestral_continuum_dataset_pmao2(
                        train_data,
                        test_data,
                        expression_data,
                        black_id,
                        white_id,
                        ancestry_dim=2,
                        ):
    # Filter train_data and test_data to only samples that have expression data
    expr_ids = set(expression_data['Samples'])
    black_set = set(black_id) if black_id is not None else None
    white_set = set(white_id) if white_id is not None else None

    filtered_train_data = {sid: vec for sid, vec in train_data.items() if sid in expr_ids}
    black_test_data  = {sid: vec for sid, vec in test_data.items()  if sid in expr_ids and (black_set is None or sid in black_set)}
    white_test_data  = {sid: vec for sid, vec in test_data.items()  if sid in expr_ids and (white_set is None or sid in white_set)}

    if len(black_test_data) == 0 or len(black_test_data) ==0 or len(white_test_data) == 0:
        raise ValueError("size of black test/white test/train set is zero")
    sample_ind = {} # a hashmap for sample names and their indices in the expression data
    for i in range(len(expression_data['Samples'])):
        sample_name = expression_data['Samples'][i]
        sample_ind[sample_name] = i 


    black_test_omics_data = {}
    white_test_omics_data = {}
    train_omics_data = {}


    # gather omics and ancestry features for white and black test samples, and all training samples
    for test_id, test_vec in white_test_data.items():
        test_genetic_feat = np.asarray(test_vec)[:ancestry_dim].reshape((1, -1)) # select a couple of pcs at the beginning
        test_omics_feat = expression_data['X'][sample_ind[test_id]].reshape((1, -1))
        test_lab = expression_data['Y'][sample_ind[test_id]]
      
        white_test_omics_data[test_id] = (test_omics_feat, test_genetic_feat, test_lab)
  
    for test_id, test_vec in black_test_data.items():
        test_genetic_feat = np.asarray(test_vec)[:ancestry_dim].reshape((1, -1)) # select a couple of pcs at the beginning
        test_omics_feat = expression_data['X'][sample_ind[test_id]].reshape((1, -1))
        test_lab = expression_data['Y'][sample_ind[test_id]]
      
        black_test_omics_data[test_id] = (test_omics_feat, test_genetic_feat, test_lab)

    train_label_candidates = set([])
    for train_id, train_vec in filtered_train_data.items():
        train_genetic_feat = np.asarray(train_vec)[:ancestry_dim].reshape((1, -1)) # select a couple of pcs at the beginning
        train_omics_feat = expression_data['X'][sample_ind[train_id]].reshape((1, -1))
        train_lab = expression_data['Y'][sample_ind[train_id]]
        train_omics_data[train_id] = (train_omics_feat, train_genetic_feat, train_lab)
        train_label_candidates.add(train_lab)
    if len(train_label_candidates) < 2:
        raise ValueError("the training label number is smaller than 2")

    return (train_omics_data,
            black_test_omics_data,
            white_test_omics_data)
# ------------ process the expression data (classification) -----------
def get_classificaition_data(dic, year):
    survivalship =  1 - dic['C']

    threshold = 365 * year
    
    selection = ~((dic["T"] < threshold) & survivalship == 1) # based on my understanding, this removes the rows with days < 365 * years and uncensored values
    X = dic['X'][selection]; 
    T = dic['T'][selection]
    Y = np.zeros(X.shape[0]); Y[T < threshold] = 1 
    Samples = dic['Samples'][selection]

    dic2 = {}
    dic2['X'] = X; dic2['Y'] = Y; dic2['Samples'] = Samples
    return dic2

# ------------ process the expression data (regression) -----------
def get_regression_data(dic):
    dic2 = {}
    dic2['X'] = dic['X']; dic2['Y'] = dic['T']; dic2['C'] = dic['C']; dic2['Samples'] = dic['Samples']
    return dic2
    
def load_expression_data(datatype, cache_dir, year = 1, target='OS', task = 'classification'):
     # ---------- Load combined cache if present ----------
    os.makedirs(cache_dir, exist_ok=True)
    if task == 'classification':
        cache_name = f"Include_cancer_info_{datatype}_year{year}_{target}.npz"
    else:
        cache_name = f"Include_cancer_info_{datatype}_regression_{target}.npz"
    cache_path = os.path.join(cache_dir, cache_name)
   
    if os.path.exists(cache_path):
        dataset = np.load(cache_path, allow_pickle=True)["results"].item()
        print(f"[CACHE HIT] Loaded year={year}, endpoint={target} from {cache_path}")
        return dataset
    
    PCA_FE_All = True # if PCA is used for feature extraction with all samples
    genders = ("MALE","FEMALE")
    groups = ( 'WHITE', 'BLACK')
    data_Category = 'R' # 'R', 'GR' ; it is 'GR' if MGtoMGF (Or) MGtoMGM = True

    if datatype == 'mRNA':
        dataset = get_mRNA(target=target,groups=groups,Gender=genders,data_Category=data_Category,
                                AE_MLTask=None, PCA_FE_All=PCA_FE_All)
    elif datatype == 'MicroRNA':
        dataset = get_MicroRNA(target=target,groups=groups,Gender=genders,data_Category=data_Category,
                                AE_MLTask=None, PCA_FE_All=PCA_FE_All)
    elif datatype == 'Methylation':
        dataset = get_Methylation(target=target,groups=groups,Gender=genders,data_Category=data_Category,
                                AE_MLTask=None, PCA_FE_All=PCA_FE_All)
    if task == 'classification':
        dataset  = get_classificaition_data(dataset, year)
    elif task == 'regression':
        dataset  = get_regression_data(dataset)
    # ---------- Save results dict ----------
    np.savez_compressed(cache_path, results=dataset)
    print(f"[CACHE SAVE] Saved results dict → {cache_path}")
    return dataset

# dimension reduction 
def feature_reduction(features_count, seed, expression_dic):
    # merge all features 
    from data_utils import data_preprocess

    feat = expression_dic['X'];
    pca_preprocess = data_preprocess(n_components=features_count, random_state=seed); 
    pca_preprocess.fit(feat)
    expression_dic['X'] = pca_preprocess.transform(feat)
    return expression_dic

# dimension reduction for the data that concatenates the feature and label
def feature_include_label_reduction(features_count, seed, expression_dic):
    # merge all features 
    from data_utils import data_preprocess

    feat = expression_dic['X'];
    lab = expression_dic['Y']
    lab = (lab - np.mean(lab)) / np.std(lab)
    feat_lab_concat = np.hstack((feat, lab.reshape((-1,1))))
    pca_preprocess = data_preprocess(n_components=features_count, random_state=seed); 
    pca_preprocess.fit(feat_lab_concat)
    expression_dic['X_Y_pca'] = pca_preprocess.transform(feat_lab_concat)
    return expression_dic

# ------------ process the demagraphic data  -----------
def get_sample_based_on_race(data_path):
    df = pd.read_csv(os.path.join(data_path, "tcga_clinical.tsv"), sep="\t")

    # Normali`ze race strings (optional but recommended)
    df["demographic.race"] = df["demographic.race"].str.strip().str.lower()

    # Filter White samples
    white_df = df[df["demographic.race"] == "white"]
    white_ids = np.unique(white_df["cases.submitter_id"].tolist())

    # Filter Black or African American samples
    black_df = df[df["demographic.race"] == "black or african american"]
    black_ids = np.unique(black_df["cases.submitter_id"].tolist())

    print("White sample count:", len(white_ids))
    print("Black sample count:", len(black_ids))

    print("Example White IDs:", white_ids[:10])
    print("Example Black IDs:", black_ids[:10])
    return white_ids, black_ids
