import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch.nn.functional import gumbel_softmax
from torch import Tensor


def mlp(in_dim, hidden_dim, out_dim, num_layers=2, p=0.1):
    layers = []
    d = in_dim
    for _ in range(num_layers - 1):
        layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(p)]
        d = hidden_dim
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class SINRUEOnlyGNN(MessagePassing):
    def __init__(self, in_ue, num_pilots, hidden=64, num_layers=3, p=0.2, eps=1e-30):
        super().__init__(aggr='add')
        self.enc_ue = mlp(in_ue, hidden, hidden, 2, p)
        self.e_uu = mlp(1, hidden, hidden, 2, p)
        self.eps = eps

        self.nn_layers = nn.ModuleList()
        self.eps_params = nn.ParameterList()
        self.lin_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()

        for _ in range(num_layers):
            self.nn_layers.append(mlp(hidden, hidden, hidden))
            self.eps_params.append(nn.Parameter(torch.zeros(1)))
            self.lin_layers.append(Linear(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))
            self.drops.append(nn.Dropout(p))

        self.pilot_head = nn.Linear(hidden, num_pilots)
        self.b_head     = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.q_head     = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self._current_lin = None

    def gin_conv(self, x, edge_index, edge_attr, nn_layer, eps_param, lin):
        self._current_lin = lin  
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = out + (1 + eps_param) * x
        return nn_layer(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        edge_attr = self._current_lin(edge_attr)
        return (x_j + edge_attr).relu()

    def forward(self, data):
        x_ue = data.x
        edge_index_uu = data.edge_index
        edge_attr_uu = data.edge_attr
        h = self.enc_ue(x_ue)
        e = self.e_uu(edge_attr_uu)

        for nn_layer, eps_param, lin, norm, drop in zip(
            self.nn_layers, self.eps_params, self.lin_layers,
            self.norms, self.drops
        ):
            h_new = self.gin_conv(h, edge_index_uu, e, nn_layer, eps_param, lin)
            h = drop(norm(h_new + h))

        pilot_logits = self.pilot_head(h)
        b = self.b_head(h)
        q = self.q_head(h)
        return pilot_logits, b, q
    

def calculate_SINR(beta, phi_matrix,deploy_param,  device, q, b):
    beta = torch.as_tensor(beta.T, dtype=torch.float32, device=device)  
    q    = torch.as_tensor(q, dtype=torch.float32, device=device)          
    b    = torch.as_tensor(b, dtype=torch.float32, device=device)          

    k_B = 1.380649e-23
    T_0 = 290.0
    NF_dB = deploy_param.NF           
    NF_lin = 10.0**(NF_dB/10.0)       
    B = deploy_param.banwidth         
    sigma_n2 = B * k_B * T_0 * NF_lin # 
    sigma_n2 = torch.tensor(sigma_n2)
    tau = deploy_param.tau            
    N   = deploy_param.N            
    P_pilot = 1                        
    rho_p = P_pilot/sigma_n2      
    rho = 1e3 / sigma_n2               

    phi_matrix = torch.as_tensor(phi_matrix, dtype=torch.float32, device=device)
    M, K = beta.shape
    c_mk_raw = tau * rho_p * b[None, :] * beta
    c_mk = torch.sqrt(c_mk_raw)
    b_beta = b[None, :] * beta
    denom_pilot = torch.matmul(b_beta, phi_matrix.T)
    denom_gamma = tau * rho_p * denom_pilot + 1.0
    gamma_mk_raw = tau * rho_p * b[None, :] * (beta ** 2) * c_mk
    gamma_mk = gamma_mk_raw/denom_gamma
    gamma_sum = gamma_mk.sum(dim=0)
    numerator = q * N * (gamma_sum ** 2)
    b_k = b.view(1, K)   
    b_kp = b.view(K, 1) 
    beta_k_expand = beta.T.unsqueeze(0)      
    beta_kp_expand = beta.T.unsqueeze(1)    
    b_k_expand = b_k.expand(K, K)
    b_kp_expand = b_kp.expand(K, K)
    sqrtfrac_num = b_kp_expand.unsqueeze(2) * beta_kp_expand
    sqrtfrac_den = b_k_expand.unsqueeze(2) * beta_k_expand
    sqrtfrac = torch.sqrt(sqrtfrac_num/sqrtfrac_den)
    gamma_expand = gamma_mk.T.unsqueeze(0)
    sum_m = (gamma_expand * sqrtfrac).sum(dim=2)
    mask = ~torch.eye(K, dtype=torch.bool, device=device)
    phi_broadcast = phi_matrix
    q_broadcast = q
    denom1 = (q_broadcast[None, :] * (sum_m ** 2) * phi_broadcast)[mask].reshape(K, K - 1).sum(dim=1)
    gamma_expand2 = gamma_mk.unsqueeze(2)
    beta_expand2 = beta.unsqueeze(1)
    gamma_beta = (gamma_expand2 * beta_expand2).sum(dim=0)
    denom2 = (q[None, :] * gamma_beta).sum(dim=1)
    denom3 = (1.0 / rho) * gamma_mk.sum(dim=0)
  
    denominator = denom1 + denom2 + denom3
    sinr_k = numerator / denominator
    sinr_k = sinr_k
    
    return sinr_k

def sinr_loss_hetero(logits_all,b_all,q_all, data,beta_batch,deploy_param,device,lambda1=0.01, tau_gumbel=1.0
):
    ue_batch = data.batch        
    B = int(ue_batch.max().item() + 1)
    tau_c = deploy_param.tau_c
    b_all = b_all.view(-1)
    q_all = q_all.view(-1)
    phi_all = gumbel_softmax(
        logits_all, tau=tau_gumbel, hard=False, dim=-1
    )                       

    losses = []
    rmin_values = []         

    for i in range(B):
        mask = (ue_batch == i)
        logits_i = logits_all[mask]      
        phi_i    = phi_all[mask]         
        b_i      = b_all[mask]           
        q_i      = q_all[mask]           
        betas = beta_batch[i].to(device) 
        phi_overlap = phi_i @ phi_i.T   
        SINR_k = calculate_SINR(
            betas, phi_overlap, deploy_param, device, q_i, b_i
        )                            

        with torch.no_grad():
            assigned = logits_i.argmax(dim=-1) 
            nuniq = torch.unique(assigned).numel()
            prelog = (tau_c - nuniq) / (tau_c)

        Rk = prelog * torch.log2(1.0 + SINR_k )  
        Rmin = Rk.min()
        rmin_values.append(Rmin.detach().item())

        term1 = torch.sum(torch.sigmoid(0.3 / (Rk + 1e-10)))
        alpha = 2.0     
        avg_rate = Rk.mean()  

        loss_i = term1 - alpha * avg_rate - lambda1 * Rmin
        losses.append(loss_i)

    return torch.stack(losses).mean()

def train(model, loader, optimizer, deploy_param, device, lambda1):
    model.train()
    total_loss = 0
    best_output = None
    min_loss = float('inf')

    for idx, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        graphs_list = data.to_data_list()
        beta_batch = torch.stack([g.beta for g in graphs_list], dim=0).to(device)  

        logits, b_out, q_out = model(data)  

        loss = sinr_loss_hetero(
            logits, b_out, q_out,
            data=data,
            beta_batch=beta_batch,
            deploy_param=deploy_param,device=device,
            lambda1=lambda1   
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()

        if loss.item() < min_loss:
            min_loss = loss.item()
            best_output = logits.detach()

    return total_loss / len(loader), best_output

def test(model, loader, deploy_param, device, lambda1):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            graphs_list = data.to_data_list()
            beta_batch = torch.stack([g.beta for g in graphs_list], dim=0).to(device)
            logits, b_out, q_out = model(data)
            loss = sinr_loss_hetero(
            logits, b_out, q_out,
            data=data,
            beta_batch=beta_batch,
            deploy_param=deploy_param,device=device, 
            lambda1=lambda1
        )
            total_loss += loss.item()
    return total_loss / len(loader)


def trainmodel(name, model, scheduler, train_loader, val_loader, optimizer, deploy_param, device):
    train_loss_hist, val_loss_hist = [], []

    lambda1 = torch.tensor(1e-2, device=device)
    for epoch in tqdm(range(1, 200), desc="Epoch"):
        train_loss, best_output_epoch = train(model, train_loader, optimizer, deploy_param, device, lambda1)
        val_loss = test(model, val_loader, deploy_param, device, lambda1)
        lambda1 = torch.clamp(lambda1 * 1.05, max=10)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        if val_loss == min(val_loss_hist):
            torch.save(model.state_dict(), f"{name}")


        print(f"Epoch {epoch:03d} | Train {train_loss:.4f} | Val {val_loss:.4f}")
        scheduler.step(val_loss)

    # Save history
    import pandas as pd
    df = pd.DataFrame({'train_loss': train_loss_hist, 'val_loss': val_loss_hist})
    df.to_csv(f'loss_history_{deploy_param.number_of_snapshots}_{deploy_param.K}_{deploy_param.L}.csv', index_label='epoch')

    return train_loss_hist, val_loss_hist

# ----------------- Evaluation -----------------
def testing(model_gnn, test_loader, deploy_param, device):
    K, _= deploy_param.K, deploy_param.L
    num_pilots = deploy_param.tau
    tau_c = deploy_param.tau_c
    model_gnn.eval()
    Rk_user_gnn,Rk_min_gnn = [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            B = data.num_graphs

            graphs_list = data.to_data_list()
            beta_batch = torch.stack([g.beta for g in graphs_list], dim=0).to(device)
            logits_gnn, b_pred_gnn, q_pred_gnn = model_gnn(data)
            phi_all_gnn = gumbel_softmax(
                logits_gnn, hard=True
            )
            phi_all_gnn = phi_all_gnn.view(B, K, num_pilots) 
            b_pred_gnn = b_pred_gnn.view(B, K)              
            q_pred_gnn = q_pred_gnn.view(B, K)              
            for i in range(B):
                betas = beta_batch[i]  
                phi_gnn = phi_all_gnn[i]      
                b_i_gnn = b_pred_gnn[i]       
                q_i_gnn = q_pred_gnn[i]       
                assigned_gnn = phi_gnn.argmax(dim=-1)
                nuniq_gnn = torch.unique(assigned_gnn).numel()
                phi_overlap_gnn = phi_gnn @ phi_gnn.T
                SINR_k_gnn = calculate_SINR(
                    betas, phi_overlap_gnn, deploy_param, device, q_i_gnn, b_i_gnn
                )
                Rk_gnn = ((tau_c - nuniq_gnn) / tau_c) * torch.log2(1.0 + SINR_k_gnn)
               
                Rk_user_gnn.extend(Rk_gnn.detach().cpu().numpy().tolist())
                Rk_min_gnn.append(Rk_gnn.min().item())
    return Rk_user_gnn, Rk_min_gnn
