import numpy as np
from joblib import Parallel, delayed
import numpy as np, torch, os, traceback
from torch_geometric.data import Data
import time
import tqdm

def _to_tensor(x):
    if isinstance(x, torch.Tensor): return x
    return torch.tensor(x, dtype=torch.float32)

def _zscore_cols(x: torch.Tensor, eps: float = 1e-6):
    return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + eps)

def _minmax01(x: torch.Tensor, eps: float = 1e-8):
    xmin = x.min(dim=0, keepdim=True).values
    xmax = x.max(dim=0, keepdim=True).values
    return (x - xmin) / (xmax - xmin + eps)


def _pairwise_dist(a, b):
    diff = a[:,None,:] - b[None,:,:]
    return torch.sqrt((diff**2).sum(dim=2) + 1e-12)

def _save_graph_cpu(graph, out_dir, m):
    
    graph = graph.to('cpu')
    path = os.path.join(out_dir, f"snap_{m:05d}.pt")
    torch.save(graph, path)
    
    return path


def RandomAPLocations(deploy_param, fixed_positions=None):
    M = deploy_param.M
    squareLength = deploy_param.squareLength
    nbrAPsPerDim = int(np.floor(np.sqrt(M))) 
    grid_size = nbrAPsPerDim ** 2
    interAPDistance = squareLength / nbrAPsPerDim
    grid_x = np.linspace(interAPDistance / 2, squareLength - interAPDistance / 2, nbrAPsPerDim)
    grid_y = np.linspace(interAPDistance / 2, squareLength - interAPDistance / 2, nbrAPsPerDim)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    grid_positions = grid_xx.flatten(order='F') + 1j * grid_yy.flatten(order='F')
    APpositions = grid_positions[:grid_size]

    if M > grid_size:
        extra = M - grid_size
        extra_positions = np.random.uniform(0, squareLength, size=(extra, 2))
        extra_positions_complex = extra_positions[:, 0] + 1j * extra_positions[:, 1]
        APpositions = np.concatenate([APpositions, extra_positions_complex])
    
    
    if fixed_positions is not None:
        APpositions = fixed_positions

    return APpositions.reshape((M, 1))

def generate_channel(deploy_param, AP_positions):
    K = deploy_param.K
    M = deploy_param.M
    f_c = deploy_param.freq_carrier
    h_AP = deploy_param.h_AP
    h_u = deploy_param.h_u
    d0, d1 = deploy_param.d0, deploy_param.d1
    sigma_sh_dB = deploy_param.sigma_sh_dB
    squareLength = deploy_param.squareLength

    user_positions = np.random.uniform(0, squareLength, size=(K, 2))
    AP_xy = np.stack([AP_positions.real.flatten(), AP_positions.imag.flatten()], axis=1)
   
    delta = user_positions[:, np.newaxis, :] - AP_xy[np.newaxis, :, :]  
    delta_wrapped = (delta + squareLength / 2) % squareLength - squareLength / 2 

    distancetoUE = np.linalg.norm(delta_wrapped, axis=2) 
    distances = np.sqrt((h_AP - h_u)**2 + distancetoUE**2) / 1000
    log_fc = np.log10(f_c)
    L_const = (46.3 + 33.9 * log_fc - 13.82 * np.log10(h_AP)
               - (1.1 * log_fc - 0.7) * h_u + (1.56 * log_fc - 0.8))

    path_loss_dB = np.zeros_like(distances)
    mask1 = distances > d1
    mask2 = (distances > d0) & (distances <= d1)
    mask3 = distances <= d0

    path_loss_dB[mask1] = L_const + 35 * np.log10(distances[mask1])
    path_loss_dB[mask2] = L_const + 15 * np.log10(d1) + 20 * np.log10(distances[mask2])
    path_loss_dB[mask3] = L_const + 15 * np.log10(d1) + 20 * np.log10(d0)
    X_dB = sigma_sh_dB * np.random.randn(K, M)      
    beta_km_dB = -path_loss_dB +X_dB

    return beta_km_dB, user_positions

def gale_shapley_matching(deploy_param, Channel_gain):
    K = deploy_param.K
    M = deploy_param.M
    capacity = deploy_param.capacity
    delta = deploy_param.delta
    S = Channel_gain.shape[0] 
    D = [{l: [] for l in range(M)} for _ in range(S)]
    for i in range(S):
        betas = Channel_gain[i] 
        user_prefs = np.argsort(betas, axis=1)[:, ::-1] 
        free_users = set(range(K))
        proposals = np.zeros(K, dtype=int)
        assigned_users = [set() for _ in range(M)]
        while free_users:
            user = next(iter(free_users))
            pref_index = proposals[user]
            if pref_index >= M:
                free_users.remove(user)
                continue
            ap = user_prefs[user, pref_index]
            proposals[user] += 1
            if len(D[i][ap]) < capacity:
                D[i][ap].append(user)
                assigned_users[ap].add(user)
                free_users.remove(user)
            else:
                min_user = min(D[i][ap], key=lambda u: betas[u, ap])
                if betas[user, ap] > betas[min_user, ap]:
                    D[i][ap].remove(min_user)
                    assigned_users[ap].remove(min_user)
                    D[i][ap].append(user)
                    assigned_users[ap].add(user)
                    free_users.remove(user)
                    free_users.add(min_user)

        for ap in range(M):
            if len(D[i][ap]) < capacity:
                remaining_capacity = capacity - len(D[i][ap])
                candidates = [k for k in range(K)
                              if betas[k, ap] >= delta and k not in assigned_users[ap]]
                candidates.sort(key=lambda k: betas[k, ap], reverse=True)
                for k in candidates[:remaining_capacity]:
                    D[i][ap].append(k)
                    assigned_users[ap].add(k)
    return D

def compute_snapshot_contamination(s, clus, gain_lin, eps=1e-30):
    gain = np.asarray(gain_lin, dtype=np.float64)
    K, M = gain.shape

    rev = {}
    for ap, users in clus.items():
        for u in users:
            rev.setdefault(int(u), []).append(int(ap))

    users_all = sorted(rev.keys())
    if len(users_all) == 0:
        return {}

    A = np.zeros((K, M), dtype=np.float64)
    for u, aps in rev.items():
        A[u, aps] = 1.0

    sum_beta = (gain * A).sum(axis=1)  

    G = gain @ A.T  

    contamination_s = {}

    for k in users_all:
        den_k = sum_beta[k] if sum_beta[k] > 0 else 1.0

        for kp in users_all:
            if kp == k:
                continue
            if (A[k] * A[kp]).sum() == 0:
                continue

            den_kp = sum_beta[kp] if sum_beta[kp] > 0 else 1.0

            num1 = G[k, kp]
            num2 = G[kp, k]

            term1 = np.log1p(num1 / (den_k + eps))
            term2 = np.log1p(num2 / (den_kp + eps))

            contamination_s[(s, k, kp)] = term1 + term2

    return contamination_s

def _normalize_cluster_format_for_snapshot(cluster_in, M, snap_idx=0):
    if isinstance(cluster_in, list) and len(cluster_in)>0 and isinstance(cluster_in[0], dict):
        return cluster_in[snap_idx]
    if isinstance(cluster_in, dict):
        return cluster_in
    if isinstance(cluster_in, list) and len(cluster_in)==M and all(isinstance(x,(list,tuple)) for x in cluster_in):
        return {ap: list(map(int, users)) for ap, users in enumerate(cluster_in)}
    raise TypeError("cluster must be a dict {ap:[ue..]} or a list-of-dicts per snapshot.")

def compute_chi_from_beta(beta_batch, eps=1e-50):
    squeeze_batch = False
    if beta_batch.dim() == 2:  
        beta_batch = beta_batch.unsqueeze(0)
        squeeze_batch = True
    elif beta_batch.dim() != 3:
        raise ValueError(f"beta_batch must be 2D or 3D, got shape {beta_batch.shape}")

    beta_sum = beta_batch.sum(dim=-1)
    eta = torch.log10(beta_sum + eps) 
    mean = eta.mean(dim=1, keepdim=True) 
    std  = eta.std(dim=1, keepdim=True) + 1e-50
    z = (eta - mean) / std               
    norm = torch.sqrt((z ** 2).sum(dim=1, keepdim=True) + eps) 
    chi = z / norm                                        

    return chi.squeeze(0) if squeeze_batch else chi


# ---------- Nodes ----------
def build_node_features(chi,
    beta_km, ue_xy=None, cluster=None, k_top: int = 3,
    include_positions: bool = True, normalize: bool = True,
):
    beta = _to_tensor(beta_km).float()  # [K,L]
    K, L = beta.shape
    k = min(k_top, L)

    ue_mean = beta.mean(dim=1, keepdim=True)
    ue_max  = beta.max(dim=1, keepdim=True).values
    ue_med  = beta.median(dim=1, keepdim=True).values
    ue_std  = beta.std(dim=1, keepdim=True, unbiased=False)
    topk_vals = torch.topk(beta, k=k, dim=1, largest=True).values
    p = beta / (beta.sum(dim=1, keepdim=True) + 1e-12)
    ue_entropy = -(p * (p + 1e-12).log()).sum(dim=1, keepdim=True)


    chi = torch.tensor(chi, dtype=torch.float32).view(-1, 1)   

    ue_cols = ["ue_mean","ue_max","ue_median","ue_std"] + \
          [f"ue_top{r}" for r in range(k,0,-1)] + \
          ["ue_entropy", "chi"]

    ue_stack = [
        ue_mean,            
        ue_max,             
        ue_med,             
        ue_std,             
        topk_vals,          
        ue_entropy,         
        chi                 
    ]

    if cluster is not None:
        ue_to_aps = [[] for _ in range(K)]
        for a, users in cluster.items():
            for u in users:
                if 0 <= u < K: ue_to_aps[u].append(a)
        ue_ap_count = torch.tensor([[len(ue_to_aps[u])] for u in range(K)], dtype=torch.float32)
        ue_beta_assigned_sum = torch.tensor(
            [[beta[u, ue_to_aps[u]].sum().item() if len(ue_to_aps[u])>0 else 0.0] for u in range(K)],
            dtype=torch.float32
        )
        ue_stack += [ue_ap_count, ue_beta_assigned_sum]
        ue_cols += ["ue_ap_count","ue_beta_assigned_sum"]

    if include_positions and ue_xy is not None:
        ue_xy_t = _to_tensor(ue_xy).float()
        ue_xy_t = _minmax01(ue_xy_t)
        ue_stack.append(ue_xy_t)
        ue_cols += ["ue_x","ue_y"]

    ue_x = torch.cat(ue_stack, dim=1)
    if normalize: ue_x = _zscore_cols(ue_x)


    return ue_x, ue_cols

# ---------- Edges ----------
def build_edges(
    beta_km, ue_xy=None, ap_xy=None, contamination=None,
      topm_conflict=8, include_distances=True
):
    beta = _to_tensor(beta_km) 
    K, _ = beta.shape

    ue_ue_u, ue_ue_v, ue_ue_attr = [], [], []
    if contamination:
        per_u = [[] for _ in range(K)]
        for (i, j), c in contamination.items():
            if 0 <= i < K and 0 <= j < K and i != j:
                per_u[i].append((j, float(c)))
        for i in range(K):
            if not per_u[i]: continue
            per_u[i].sort(key=lambda x: -x[1])
            for j, c in per_u[i][:topm_conflict]:
                ue_ue_u.append(i); ue_ue_v.append(j)
                ue_ue_attr.append([c])
    ue_ue_edge_index = torch.tensor([ue_ue_u, ue_ue_v], dtype=torch.long) if ue_ue_u else torch.zeros((2,0), dtype=torch.long)
    ue_ue_edge_attr  = torch.tensor(ue_ue_attr, dtype=torch.float32) if ue_ue_attr else torch.zeros((0,1), dtype=torch.float32)

    if include_distances and (ue_xy is not None) and len(ue_ue_u) > 0:
        ue_xy_t = _to_tensor(ue_xy)
        d_list = []
        for i, j in zip(ue_ue_u, ue_ue_v):
            dx = ue_xy_t[i] - ue_xy_t[j]
            d_list.append([torch.sqrt((dx*dx).sum()).item()])
        d_col = torch.tensor(d_list, dtype=torch.float32)
        ue_ue_edge_attr = torch.cat([ue_ue_edge_attr, d_col], dim=1)

    return ue_ue_edge_index, ue_ue_edge_attr

# ---------- Graph builder ----------

def make_graph_one_snapshot(chi,
    beta_km, ue_xy, ap_xy, cluster, contamination_dict, topm_conflict=8,
    include_positions=True, include_distances=True
):
    
    ue_x, _ = build_node_features(chi,
        beta_km=beta_km, ue_xy=ue_xy if include_positions else None,
        cluster=cluster, k_top=3, normalize=True
    )
    ue_ue_ei, ue_ue_ea = build_edges(
        beta_km=beta_km, ue_xy=ue_xy if include_distances else None,
        ap_xy=ap_xy if include_distances else None,
        contamination=contamination_dict, topm_conflict=topm_conflict,
        include_distances=include_distances
    )
    
    data = Data(
        x=ue_x, 
        edge_index=ue_ue_ei, 
        edge_attr=ue_ue_ea
    )
    
    return data

def  worker_batch(start_idx, end_idx, deploy_param_dict, APpositions_complex,
                 ap_capacity, topk_ap, topm_conflict,
                 include_positions, include_distances,
                 out_dir, base_seed):

    os.makedirs(out_dir, exist_ok=True)
    paths, errors = [], []

    class _DP: pass
    dp = _DP()
    for k,v in deploy_param_dict.items(): setattr(dp, k, v)
    K, M = dp.K, dp.M

    ap_xy = torch.stack([
        torch.tensor(APpositions_complex.real.flatten(), dtype=torch.float32),
        torch.tensor(APpositions_complex.imag.flatten(), dtype=torch.float32)
    ], dim=1)

    for s in range(start_idx, end_idx):
        try:
            np.random.seed(base_seed + s)
            torch.manual_seed(base_seed + s)

            beta_km_np, ue_xy_np = generate_channel(dp, APpositions_complex)
            beta_km_np = np.asarray(beta_km_np)
            if beta_km_np.shape != (K, M):
                raise ValueError(f"[s={s}] Expected (K,M)=({K},{M}), got {beta_km_np.shape}")

            ue_xy2d_t = torch.as_tensor(ue_xy_np, dtype=torch.float32)

            beta_np = np.asarray(beta_km_np)
            if beta_np.max() <= 0.0:
                beta_np = 10.0 ** (beta_np / 10.0)

            beta_lin_t = torch.tensor(beta_np, dtype=torch.float32, device=ap_xy.device)
            chi = compute_chi_from_beta(beta_lin_t.unsqueeze(0)).squeeze(0)
           
            D = gale_shapley_matching(dp, beta_km_np[None, :, :])
            cluster_s = _normalize_cluster_format_for_snapshot(D, M=M, snap_idx=0)

            
            contam_all = compute_snapshot_contamination(0, cluster_s, beta_lin_t)
        
            if len(contam_all) > 0:
                k0 = next(iter(contam_all))
                if isinstance(k0, tuple) and len(k0) == 3:
                    contam_s = {(i,j): float(v) for (mm,i,j), v in contam_all.items() if mm == 0}
                else:
                    contam_s = {(i,j): float(v) for (i,j), v in contam_all.items()}
            else:
                contam_s = {}

           
            g = make_graph_one_snapshot(chi, 
                beta_km=beta_lin_t, ue_xy=ue_xy2d_t, ap_xy=ap_xy,
                cluster=cluster_s, contamination_dict=contam_s,topm_conflict=topm_conflict,
                include_positions=include_positions, include_distances=include_distances
            )
            beta_np = np.asarray(beta_km_np)
            if beta_np.max() <= 0.0:
                beta_np = 10.0 ** (beta_np / 10.0)
            
            g.beta = torch.tensor(beta_np, dtype=torch.float32, device=ap_xy.device) 

        
            path = _save_graph_cpu(g, out_dir, s)  
            paths.append(path)

        except Exception:
            err_path = os.path.join(out_dir, f"error_{s:05d}.log")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            errors.append(err_path)

    return {"paths": paths, "errors": errors}

def build_dataset_batched(deploy_param, n_snapshots=500, batch_size=None,
                          ap_capacity=9, topk_ap=9, topm_conflict=8,
                          include_positions=True, include_distances=True,
                          out_dir="graphs_tmp_batched", final_path="dataset_batched.pt",
                          n_jobs=None, base_seed=2025, verbose=10):
    os.makedirs(out_dir, exist_ok=True) 
    APpositions_complex = RandomAPLocations(deploy_param)
    deploy_param_dict = {k: getattr(deploy_param, k) for k in dir(deploy_param)
                         if not k.startswith("_") and not callable(getattr(deploy_param, k))}
 
    if n_jobs is None:
        import os as _os
        n_jobs = max(1, _os.cpu_count() - 1)

    if batch_size is None:
        batch_size = n_snapshots
        n_jobs = 1  
 
    ranges = []
    cur = 0
    while cur < n_snapshots:
        end = min(cur + batch_size, n_snapshots)
        ranges.append((cur, end))
        cur = end
    
    build_start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(
        delayed(worker_batch)(
            start, end, deploy_param_dict, APpositions_complex,
            ap_capacity, topk_ap, topm_conflict,
            include_positions, include_distances,
            out_dir, base_seed
        )

        for (start, end) in tqdm.tqdm(ranges, desc="worker_batch")
    )
    build_elapsed = time.perf_counter() - build_start
    per_snapshot = build_elapsed / max(1, n_snapshots)
    print(f"[Timing] build_dataset_batched completed in {build_elapsed:.2f} seconds "
      f"(~{build_elapsed/60:.2f} minutes, {per_snapshot:.3f} seconds/snapshot)")
    
    paths, errors = [], []
    for res in results:
        paths.extend(res["paths"])
        errors.extend(res["errors"])

    if errors:
        print(f"Ada {len(errors)} snapshot gagal. Contoh log:")
        for p in errors[:10]:
            print("  -", p)


    def _m_from_path(p):
        base = os.path.basename(p)
        num = base.replace("snap_", "").replace(".pt", "")
        return int(num)
    paths.sort(key=_m_from_path)

    graphs = [torch.load(p, weights_only=False) for p in paths]
    torch.save(graphs, final_path)
    print(f"Built {len(graphs)}/{n_snapshots} graphs → {final_path} (n_jobs={n_jobs}, batch_size={batch_size})")
    return final_path

class init_parameters:
    def __init__(self, number_UE, number_AP, num_snapshots):
        self.freq_carrier = 1.9e9/1e6
        self.h_AP = 30
        self.h_u = 1.65
        self.d0 = 0.01
        self.d1 = 0.05 
        self.K = number_UE
        self.M = number_AP
        self.squareLength = 1000
        self.number_of_snapshots = num_snapshots
        self.capacity = 9
        self.delta = -120
        self.N = 4
        self.sigma_sh_dB = 6
        self.banwidth = 20e6
        self.T0 = 290
        self.tau_c = 200

if __name__ == "__main__":
    number_UE = 64
    number_AP = 128
    number_snapshots = 5000
    deploy_param = init_parameters(number_UE, number_AP,number_snapshots)
    out = build_dataset_batched(
        deploy_param=deploy_param,
        n_snapshots=number_snapshots,
        batch_size=32,        
        ap_capacity=9,
        topk_ap=3,
        topm_conflict=8,
        include_positions=None,
        include_distances=None,
        out_dir="graphs_tmp_batched",
        final_path= f"dataset_batched_{number_AP}_{number_UE}_{number_snapshots}_chi.pt",
        n_jobs=1,            
        base_seed=2025,
        verbose=2
    )
    graphs = torch.load(out, weights_only=False)
    print("Total graphs:", len(graphs))
    # example
    print(graphs[0])
