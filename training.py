import os; os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from torch_geometric.loader import DataLoader
import system
from torch.optim.lr_scheduler import ReduceLROnPlateau

class init_parameters:
    def __init__(self, number_UE, number_AP, number_pilots):
        self.K = number_UE
        self.M = number_AP
        self.tau = number_pilots
        self.N = 4
        self.banwidth = 20e6
        self.NF = 10
        self.tau_c = 200

def save_graph_dataset(graph_list, filename):
    torch.save(graph_list, filename)

def load_graph_dataset(filename):
    return torch.load(filename, weights_only=False)


def train_model_graph(number_snapshots, number_UE, number_AP, number_pilots, device, save_model_path):
    param = init_parameters(number_UE, number_AP, number_pilots)
    cache_file = f"hetero_dataset_batched_{number_AP}_{number_UE}_{number_snapshots}_chi.pt"
   
    print(f"[CACHE] Loading cached graph dataset from {cache_file}...")
    graph_list = load_graph_dataset(cache_file)
    
   
    # === Split Dataset ===
    split_idx = int(0.8 * number_snapshots)
    train_graphs = graph_list[:split_idx]
    val_graphs   = graph_list[split_idx:]
    g = graph_list[0]

    # === DataLoaders ===
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=32, shuffle=False)
    # === Inspect sample ===
    sample = train_graphs[0]
    in_ue = sample.x.size(1)

    from system import SINRUEOnlyGNN
    model = SINRUEOnlyGNN(in_ue=in_ue, num_pilots=param.tau,
                    hidden=64, num_layers=3, p=0.1).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-5)

    print("[TRAIN] Start training...")
    loss_train, loss_val = system.trainmodel(save_model_path, model, scheduler, train_loader, val_loader, optimizer, param, device)

    # === Plot Loss Curve ===
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_val, label='Validation Loss')
    plt.plot(loss_train, label='Training Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training and Validation Loss')
    plt.grid(True); plt.legend()
    plt.savefig(f"loss_plot{number_UE}.png", dpi=300)
    print("[TRAIN] Loss plot saved as loss_plot.png")

import argparse
def main():
    parser = argparse.ArgumentParser(description="Train GNN model for Massive MIMO pilot assignment.")
    parser.add_argument('--number_UE', type=int, default=64, help='Number of User Equipments (UE)')
    parser.add_argument('--number_AP', type=int, default=128, help='Number of Access Points (AP)')
    parser.add_argument('--number_snapshots', type=int, default=5000, help='Number of snapshots')
    parser.add_argument('--number_pilots', type=int, default=16, help='Number of pilot sequences')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run (cpu/cuda)')
    parser.add_argument('--save_model_path', type=str, default='', help='Path to save the model')

    args = parser.parse_args()

    # If save_model_path not specified, make a default name
    if args.save_model_path == '':
        args.save_model_path = f"GIN_graph_{args.number_UE}UE_{args.number_AP}AP_{args.number_pilots}tau_{args.number_snapshots}.pt"
    print(f"Training with {args.number_UE} UE, {args.number_AP} AP, {args.number_snapshots} snapshots, {args.number_pilots} pilots")
    print(f"Model will be saved to: {args.save_model_path}")

    train_model_graph(
        number_snapshots=args.number_snapshots,
        number_UE=args.number_UE,
        number_AP=args.number_AP,
        number_pilots=args.number_pilots,
        device = torch.device(args.device),
        save_model_path=args.save_model_path, 
    )

if __name__ == "__main__":
    main()