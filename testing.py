import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
import system
import pandas as pd


class InitParameters:
    def __init__(self, number_UE, number_AP, number_pilots):
        self.K = number_UE
        self.M = number_AP
        self.tau = number_pilots
        self.number_pilots = number_pilots 
        self.N = 4
        self.banwidth = 20e6
        self.NF = 10
        self.tau_c = 200
        self.t_min = 0
        self.t_max = 1000


def plot_cdf(data, label, color, linestyle='-', linewidth=2, alpha=1.0, zorder=1):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, list):
        data = np.array(data)

    data = np.asarray(data).ravel()
    print(f"[CDF] {label}: n={len(data)}")

    if len(data) == 0:
        print(f"[Warning] Data '{label}' null. Skipping CDF plot.")
        return

    sorted_data = np.sort(data)
    cdf = np.linspace(0, 1, len(sorted_data))
    plt.plot(sorted_data, cdf, label=label, color=color,
             linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=zorder)


def load_graph_dataset(filename):
    return torch.load(filename, weights_only=False)


def save_Rk_to_csv(Rk_list, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    Rk_arr = np.asarray(Rk_list).ravel()
    pd.DataFrame({'Rk_values': Rk_arr}).to_csv(filename, index=False)
    print(f'[SAVE] {filename}  (n={len(Rk_arr)})')


def load_Rk_from_csv(kind, method, K, L, tau, dataset_snapshot):
    kind_map = {
        'user': f"Rk/Rk_user_{method}_K{K}_L{L}_tau{tau}_{dataset_snapshot}.csv",
        'min':  f"Rk/Rk_min_{method}_K{K}_L{L}_tau{tau}_{dataset_snapshot}.csv",
    }
    if kind not in kind_map:
        raise ValueError("kind must be 'user' or 'min'")
    fname = kind_map[kind]
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)
    return pd.read_csv(fname)["Rk_values"].to_numpy()


def summarize_Rk(arr):
    arr = np.asarray(arr).ravel()
    return {
        "p05": np.quantile(arr, 0.05),
        "max": np.max(arr),
    }


def _plot_and_save_cdf(res_dict, xlabel, save_path, colors=None):
    if colors is None:
        colors = plt.cm.tab10.colors

    plt.figure(figsize=(8, 6))
    for i, (label, arr) in enumerate(res_dict.items()):
        plot_cdf(arr, label=label, color=colors[i % len(colors)])

    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel('CDF', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def test_model_graph(model_GIN_path, dataset_snapshot,
                     number_UE, number_AP, number_pilots, device, model, data_path):
    # --- Init parameter ---
    param = InitParameters(number_UE, number_AP, number_pilots)

    cache_file = os.path.join(
        data_path)

    print(f"[CACHE] Loading graph dataset from:\n  {cache_file}")
    graph_list = load_graph_dataset(cache_file)

    split_idx = max(1, int(0.2 * dataset_snapshot))
    graph_list = graph_list[:split_idx]
    print(f"[DATA] Jumlah graph untuk testing: {len(graph_list)}")

    test_loader = DataLoader(graph_list, batch_size=32, shuffle=False)
    sample = graph_list[0]
    in_ue = sample.x.size(1)

    # --- Load model ---
    print(f"[MODEL] Loading GINEConv dari:\n  {model_GIN_path}")
    from system import SINRUEOnlyGNN
    model_GIN = SINRUEOnlyGNN(
        in_ue=in_ue,
        num_pilots=param.number_pilots,
        hidden=64,
        num_layers=3,
        p=0.1
    ).to(device)
    model_GIN.load_state_dict(torch.load(model_GIN_path, map_location=device),strict=False)
    model_GIN.eval()

    # --- Testing ---
    Rk_user_GIN, Rk_min_GIN = system.testing(model_GIN, test_loader, param, device)

    res      = {'GIN': np.asarray(Rk_user_GIN).ravel()}
    res_min  = {'GIN': np.asarray(Rk_min_GIN).ravel()}

    # --- Summary ---
    print("\n[SUMMARY] Per-user SE:")
    for name, arr in res.items():
        s = summarize_Rk(arr)
        print(f"  {name:8s} | p5={s['p05']:.3f}  max={s['max']:.3f}")

    print("\n[SUMMARY] Minimum SE per snapshot:")
    for name, arr in res_min.items():
        s = summarize_Rk(arr)
        print(f"  {name:8s} | p5={s['p05']:.3f}  max={s['max']:.3f}")

    # --- Plot CDF ---
    tag = f"UE{number_UE}_AP{number_AP}_tau{number_pilots}"

    _plot_and_save_cdf(
        res,
        xlabel='Spectral Efficiency per UE (bit/s/Hz)',
        save_path=f"cdf_all_{tag}.png"
    )
    _plot_and_save_cdf(
        res_min,
        xlabel='Minimum Spectral Efficiency per snapshot (bit/s/Hz)',
        save_path=f"cdf_min_{tag}.png"
    )

    # --- Save CSV ---
    fname_Rk   = f"Rk_user_{model}_K{number_UE}_M{number_AP}_tau{number_pilots}_{dataset_snapshot}.csv"
    fname_Rmin = f"Rk_min_{model}_K{number_UE}_M{number_AP}_tau{number_pilots}_{dataset_snapshot}.csv"
    save_Rk_to_csv(res['GIN'],     fname_Rk)
    save_Rk_to_csv(res_min['GIN'], fname_Rmin)

    return res, res_min

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  type=str,   required=True)
    parser.add_argument('--data_path',   type=str,   required=True)
    parser.add_argument('--snapshots',   type=int,   default=5000)
    parser.add_argument('--K',           type=int,   default=64)
    parser.add_argument('--M',           type=int,   default=128)
    parser.add_argument('--tau',         type=int,   default=16)
    parser.add_argument('--model_name',  type=str,   default='GIN')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_model_graph(
        model_GIN_path   = args.model_path,
        dataset_snapshot = args.snapshots,
        number_UE        = args.K,
        number_AP        = args.M,
        number_pilots    = args.tau,
        device           = device,
        model            = args.model_name,
        data_path        = args.data_path
    )