import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from tabpfn import TabPFNClassifier


def _fit_tabpfn(X: np.ndarray, y: np.ndarray) -> TabPFNClassifier:
    clf = TabPFNClassifier()  # you can expose args if needed
    return clf.fit(X, y)

def develop_multimodal_base_models(X_multimodal_train, y_multimodal_train):
    # This function will develop multimodal base models
    model_list = []
    for X, y in zip(X_multimodal_train, y_multimodal_train):
        model_list.append(_fit_tabpfn(X, y))
            # Further fine-tuning can be done here if needed
    return model_list

def search_mst_context(A: np.ndarray, B: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Build an MST over X = [A; B] and return indices in A that are incident to
    at least one MST edge that connects a node from A to a node from B.

    Parameters
    ----------
    A : (n_a, d) array
    B : (n_b, d) array
    metric : str
        Distance metric for pairwise_distances (e.g., 'euclidean', 'cosine', ...)

    Returns
    -------
    indices : np.ndarray[int]
        Sorted unique indices into A that touch at least one A--B edge in the MST.
    """
    n_a = len(A)
    n_b = len(B)
    if n_a == 0 or n_b == 0:
        return np.array([], dtype=int)

    X = np.vstack([A, B])
    n = len(X)

    # ---- Dense distance matrix (no SciPy dependency) ----
    D = pairwise_distances(X, metric=metric)
    # Never pick self-edges
    np.fill_diagonal(D, np.inf)

    # ---- Prim's algorithm (O(n^2)) to compute MST over complete graph with weights D ----
    in_mst = np.zeros(n, dtype=bool)
    parent = -np.ones(n, dtype=int)
    min_w = np.full(n, np.inf)

    # start from node 0
    in_mst[0] = True
    min_w[:] = D[0]
    parent[:] = 0
    parent[0] = -1

    for _ in range(n - 1):
        # pick the non-MST node with smallest connecting edge
        j = np.argmin(np.where(in_mst, np.inf, min_w))
        if np.isinf(min_w[j]):
            # graph disconnected under the metric (shouldn't happen for complete graph),
            # but guard just in case
            break
        in_mst[j] = True

        # relax edges from j
        # new candidates are nodes not yet in MST
        not_in = ~in_mst
        # update where this edge is better
        better = D[j] < min_w
        update = not_in & better
        min_w[update] = D[j, update]
        parent[update] = j

    # ---- Collect A--B cut-edges from MST ----
    # MST edges are (parent[v], v) for v with parent[v] != -1
    ab_touching_A_indices = []
    for v in range(n):
        u = parent[v]
        if u == -1:
            continue
        a_u, a_v = (u < n_a), (v < n_a)   # whether endpoints are in A
        # edge crosses the A/B cut if exactly one endpoint is in A
        if a_u != a_v:
            # add the endpoint that lies in A
            ab_touching_A_indices.append(u if a_u else v)

    return np.array(sorted(set(ab_touching_A_indices)), dtype=int)

def search_knn_context(A: np.ndarray, B: np.ndarray,  k: int = 1) -> np.ndarray:
    """
    Return indices in A whose nearest neighbors include points from B.
    These correspond to A-nodes incident to cut-edges in the k-NN graph.

    Parameters
    ----------
    A : (n_a, d) array
    B : (n_b, d) array
    k : int, default=10
        Number of neighbors (excluding self).

    Returns
    -------
    indices : np.ndarray
        Sorted array of indices into A that touch at least one cut-edge.
    """

    n_a = A.shape[0]
    X = np.vstack([A, B])

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)))
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    targets = []
    for i in range(n_a):  # only rows belonging to A
        # drop self if present
        nbrs = [j for j in indices[i] if j != i]
        # check if any neighbor is in B
        if any(j >= n_a for j in nbrs):
            targets.append(i)

    return np.array(targets)

def batched_search_context_cls(all_tr_x, all_tr_y, tr_x, tr_y, distance_metric='euclidean'):
    """
    Partition all_tr_x/all_tr_y into chunks of size len(tr_x),
    run search_context_cls on each chunk, and return indices
    w.r.t. all_tr_x.
    """
    n_total = len(all_tr_x)
    batch_size = len(tr_x)

    global_indices = []

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)

        # slice the current partition
        part_x = all_tr_x[start:end]
        part_y = all_tr_y[start:end]

        # search_context_cls should return indices relative to part_x/part_y
        local_inds = search_context_cls(part_x, part_y, tr_x, tr_y, distance_metric=distance_metric)

        # map local indices back to global indices of all_tr_x
        for i in local_inds:
            global_indices.append(start + i)

    return global_indices  # or np.array(global_indices) if you prefer

# a context enhancement scheme that only includes the similar data in terms of p(x, y)
def search_context_cls(
    A: np.ndarray, y_A: np.ndarray,
    B: np.ndarray, y_B: np.ndarray,
    distance_metric: str = 'euclidean',
) -> np.ndarray:
    # ind_search_rel = search_knn_context(A, B, k=k)
    ind_search_rel = search_mst_context(A, B, metric=distance_metric)

    return ind_search_rel


def batched_search_context_cls_by_class(
    all_tr_x, all_tr_y, tr_x, tr_y, distance_metric='euclidean', batch_size=None
):
    selected = []
    for cls in np.unique(np.asarray(tr_y).reshape(-1)):
        src_mask = np.asarray(all_tr_y).reshape(-1) == cls
        tgt_mask = np.asarray(tr_y).reshape(-1) == cls
        if not np.any(src_mask) or not np.any(tgt_mask):
            continue
        local = batched_search_context_cls(
            np.asarray(all_tr_x)[src_mask],
            np.asarray(all_tr_y)[src_mask],
            np.asarray(tr_x)[tgt_mask],
            np.asarray(tr_y)[tgt_mask],
            distance_metric=distance_metric,
        )
        selected.extend(np.where(src_mask)[0][np.asarray(local, dtype=int)].tolist())
    return np.array(sorted(set(selected)), dtype=int)


def search_context_cls_by_class(
    A: np.ndarray, y_A: np.ndarray, B: np.ndarray, y_B: np.ndarray, distance_metric: str = 'euclidean'
) -> np.ndarray:
    selected = []
    for cls in np.unique(np.asarray(y_B).reshape(-1)):
        src_mask = np.asarray(y_A).reshape(-1) == cls
        tgt_mask = np.asarray(y_B).reshape(-1) == cls
        if not np.any(src_mask) or not np.any(tgt_mask):
            continue
        local = search_context_cls(
            np.asarray(A)[src_mask],
            np.asarray(y_A)[src_mask],
            np.asarray(B)[tgt_mask],
            np.asarray(y_B)[tgt_mask],
            distance_metric=distance_metric,
        )
        selected.extend(np.where(src_mask)[0][np.asarray(local, dtype=int)].tolist())
    return np.array(sorted(set(selected)), dtype=int)


def knn_graph_search_context_cls(
    A: np.ndarray,
    y_A: np.ndarray,
    B: np.ndarray,
    y_B: np.ndarray,
    k: int = 10,
    distance_metric: str = 'euclidean',
) -> np.ndarray:
    return np.asarray(search_knn_context(A, B, k=k), dtype=int)


def scaled_radius_candidates(feat_dim: int):
    base = np.sqrt(max(1, feat_dim))
    return np.asarray([0.25 * base, 0.5 * base, 1.0 * base, 1.5 * base], dtype=float)


def radius_search_context_cls(
    A: np.ndarray,
    y_A: np.ndarray,
    B: np.ndarray,
    y_B: np.ndarray,
    radius: float,
    distance_metric: str = 'euclidean',
) -> np.ndarray:
    nbrs = NearestNeighbors(radius=radius, metric=distance_metric)
    nbrs.fit(A)
    neighbor_ids = nbrs.radius_neighbors(B, return_distance=False)
    if len(neighbor_ids) == 0:
        return np.array([], dtype=int)
    merged = sorted({idx for row in neighbor_ids for idx in row})
    return np.asarray(merged, dtype=int)


def radius_search_context_cls_with_param_search(
    A: np.ndarray,
    y_A: np.ndarray,
    B: np.ndarray,
    y_B: np.ndarray,
    radii,
    distance_metric: str = 'euclidean',
):
    radii = list(radii)
    if not radii:
        return np.array([], dtype=int)

    best = None
    best_score = None
    for radius in radii:
        candidate = radius_search_context_cls(
            A, y_A, B, y_B, radius=radius, distance_metric=distance_metric
        )
        score = len(candidate)
        if best is None or score > best_score:
            best = candidate
            best_score = score
    return np.asarray(best, dtype=int)


def cont_radius_search_context_cls_with_param_search(
    train_original_X,
    train_original_y,
    te_context_X_list,
    te_context_y_list,
    train_core_X,
    train_core_y,
    val_context_X_list,
    val_context_y_list,
    val_X,
    val_y,
    radii,
    distance_metric='euclidean',
):
    radii = list(radii)
    if not radii:
        return [], None
    chosen_radius = radii[0]
    similar_ind_list = [
        radius_search_context_cls(
            train_original_X,
            train_original_y,
            ctx_X,
            ctx_y,
            radius=chosen_radius,
            distance_metric=distance_metric,
        )
        for ctx_X, ctx_y in zip(te_context_X_list, te_context_y_list)
    ]
    return similar_ind_list, chosen_radius


def cont_radius_search_context_cls_with_param_search_by_class(
    train_original_X,
    train_original_y,
    te_context_X_list,
    te_context_y_list,
    train_core_X,
    train_core_y,
    val_context_X_list,
    val_context_y_list,
    val_X,
    val_y,
    radii,
    distance_metric='euclidean',
):
    radii = list(radii)
    if not radii:
        return [], None
    chosen_radius = radii[0]
    similar_ind_list = [
        _radius_search_context_cls_by_class(
            train_original_X,
            train_original_y,
            ctx_X,
            ctx_y,
            radius=chosen_radius,
            distance_metric=distance_metric,
        )
        for ctx_X, ctx_y in zip(te_context_X_list, te_context_y_list)
    ]
    return similar_ind_list, chosen_radius


def _radius_search_context_cls_by_class(
    A: np.ndarray,
    y_A: np.ndarray,
    B: np.ndarray,
    y_B: np.ndarray,
    radius: float,
    distance_metric: str = 'euclidean',
) -> np.ndarray:
    selected = []
    for cls in np.unique(np.asarray(y_B).reshape(-1)):
        src_mask = np.asarray(y_A).reshape(-1) == cls
        tgt_mask = np.asarray(y_B).reshape(-1) == cls
        if not np.any(src_mask) or not np.any(tgt_mask):
            continue
        local = radius_search_context_cls(
            np.asarray(A)[src_mask],
            np.asarray(y_A)[src_mask],
            np.asarray(B)[tgt_mask],
            np.asarray(y_B)[tgt_mask],
            radius=radius,
            distance_metric=distance_metric,
        )
        selected.extend(np.where(src_mask)[0][np.asarray(local, dtype=int)].tolist())
    return np.array(sorted(set(selected)), dtype=int)

# a context enhancement scheme that includes unexplored source data and
# similar samples in the feature space only.
def search_context_cls2(
    A: np.ndarray, y_A: np.ndarray,
    B: np.ndarray, y_B: np.ndarray,
    k: int = 1, random_state: int | None = None
) -> np.ndarray:
    # include unexplored data from the source domain
    ind_exclude_candidate = search_mst_context(A, B)
    ind_unexplore = np.array(list(set(np.arange(len(A))) - set(ind_exclude_candidate)))

     # include unexplored data from the source domain
    ind_similar_joint_x_y = search_mst_context(A[ind_exclude_candidate], B)
    ind_search_rel = np.hstack((ind_unexplore, ind_exclude_candidate[ind_similar_joint_x_y]))

    return ind_search_rel


# knn classifiers 
def search_context_cls3(
    A: np.ndarray, y_A: np.ndarray,
    B: np.ndarray, y_B: np.ndarray,
    k: int = 1, 
) -> np.ndarray:
    # Fit k-NN classifier on A ∪ B
    clf = KNeighborsClassifier(n_neighbors=k)
    X_all = np.vstack((A, B))
    y_all = np.hstack((np.zeros(len(A)), np.ones(len(B))))
    clf.fit(X_all, y_all)

    # Predict labels for A (re-extract first |A| samples)
    X_A = X_all[:len(A)]
    y_pred = clf.predict(X_A)

    # Return indices of A classified as class B (i.e., y_pred == 1 or whatever)
    ind_search_rel = np.where(y_pred == 1)[0]  # Adjust label if needed
    

    return ind_search_rel
