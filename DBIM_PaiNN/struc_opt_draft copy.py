
import torch
from torch_geometric.loader import DataLoader
from PaiNN.painn_model import PaiNN
import torch.nn.functional as F
from airp import make_noise_schedule, forward_transition, predict_DBIM, predict_PaiNN, sampling
import torch
import torch.nn.functional as F
import torch.nn as nn

from loss_functions import DBIMLoss, PaiNNLoss
from airp import make_noise_schedule, forward_transition, predict_DBIM, predict_PaiNN, sampling, data_prepare
from DBIM_argument import parse_opt
from utils import read_list, load_model, get_hist, sub_center
from PaiNN.painn_model import PaiNN
from DBIM_models import DBIMGenerativeModel
from torch_scatter import scatter_add

from torch_geometric.data import Data

# @torch.no_grad()
def energy_and_grad(data, node_mask, pretrained_PaiNN, pos, args):

    pred_PaiNN, data_PaiNN = predict_PaiNN(data, node_mask, pretrained_PaiNN, args, pos=pos, use_rd=True)
    return pred_PaiNN['energy'].to(device), pred_PaiNN['forces'].to(device)     
    # return pred_PaiNN['energy']
    # return pred_PaiNN['energy'], data_PaiNN.energy_grad


def optimize_pos(model, val_loader, args, batch=None, lr=1e-5, steps=1000,
                 force_thresh=0.02, print_every=50):
    device = args.device
    dtype = torch.float32

    model.eval()
    total_dif = 0

    for batch_idx, data in enumerate(val_loader):
        pos0, _, node_mask, _ = data_prepare(data=data, args=args)
        pos = data['rdkit_pos'].to(device)[node_mask.view(-1)].detach().clone().requires_grad_(True)

        opt = torch.optim.Adam([pos], lr=lr)
        # opt = torch.optim.LBFGS([pos], lr=10.0, max_iter=20, tolerance_grad=1e-5)
        E_prev, grad_prev = energy_and_grad(data, node_mask, model, pos, args)


        maxF = (-grad_prev).norm(dim=-1).max().item()
        print(f"[0] E={E_prev.item():.12f} eV, max|F|={maxF:.12f} eV/Å")
        # pos_dif = F.l1_loss(sub_center(pos), pos0[node_mask.squeeze(-1)])
        # print(pos_dif) 

        init_patience = patience = 20
        E_curr = float('inf')
        batch_dif = 0

        for t in range(1, steps+1):
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                # pos = sub_center(pos)
                E, dE_dR = energy_and_grad(data, node_mask, model, pos, args)
                pos.grad = (-1 * dE_dR).clone()     # 最小化 E
            opt.step()

            # E = opt.step(closure)

            if E > E_curr:
                patience -= 1
            else:
                patience = init_patience
                E_curr = E

            if patience < 0:
                pos_dif = F.l1_loss(pos, pos0[node_mask.squeeze(-1)])
                batch_dif += pos_dif
                pos_dif = F.l1_loss(sub_center(pos), pos0[node_mask.squeeze(-1)])
                print(pos_dif)  
                break

            if t % 1 == 0:
                with torch.no_grad():
                    E2, dE_dR2 = energy_and_grad(data, node_mask, model, pos, args)
                    maxF = (-dE_dR2).norm(dim=-1).max().item()
                    print(f"[{t}] E={E2.item():.12f} eV, max|F|={maxF:.12f} eV/Å")
                    pos_dif = F.l1_loss(sub_center(pos), pos0[node_mask.squeeze(-1)])
                    # print(pos_dif)  
                # if maxF < force_thresh:
                #     print(f"Converged at step {t}")
                #     break

    return pos.detach().cpu()

def optimize_lbfgs_energy(model, data, args, z=None, batch=None,
                        max_outer=5, gtol=1e-3):
    pos0, _, node_mask, _ = data_prepare(data=data, args=args)
    R = data['rdkit_pos'].to(device)[node_mask.view(-1)].to(device).detach().clone()
    R.requires_grad_(True)

    def f(Rin):
        out = model(make_data_with_pos(Rin))    # 和你 closure 一样的前向
        return out['energy'].sum()

    E1 = f(R)
    E1.backward(retain_graph=True)              # 第一次 backward
    R.grad.zero_()

    E2 = f(R)
    E2.backward()                               # 第二次 backward → 若模型内部有 in-place，这里会炸
    return 0,0

    # pos0, _, node_mask, _ = data_prepare(data=data, args=args)
    # R = data['rdkit_pos'].to(device)[node_mask.view(-1)].detach().clone().requires_grad_(True)

    # data_PaiNN = Data(
    #     pos=R,
    #     batch=data.batch.to(device)[node_mask.view(-1)].detach().clone(),
    #     atomic_numbers=data.atomic_numbers.to(device)[node_mask.view(-1)].detach().clone(),
    #     energy=data.energy.to(device).detach().clone(),
    #     energy_grad=data.energy_grad.to(device)[node_mask.view(-1)].detach().clone(),
    #     npa_charges=data.npa_charges.to(device)[node_mask.view(-1)].detach().clone(),
    #     natoms=data.natoms.to(device).detach().clone(),
    #     species=data.species.to(device).to(torch.long)[node_mask.view(-1)].detach().clone()
    # )

    # opt = torch.optim.LBFGS([R], lr=1.0, max_iter=20, tolerance_grad=1e-12)

    # def closure():
    #     opt.zero_grad()
    #     # E = energy_and_grad(data, node_mask, model, R, args).sum()
    #     E = model(data_PaiNN)['energy'].sum()
    #     E.backward()                      # dE/dR 存到 R.grad（= +∇E）
    #     return E

    # for _ in range(max_outer):
    #     E = opt.step(closure)  # LBFGS 内部会多次 closure/backward
    #     with torch.no_grad():
    #         F = -R.grad
    #         fmax = F.view(-1,3).norm(dim=1).max().item()
    #         R.grad.zero_()
    #         if fmax < gtol:
    #             break

    # return float(E), fmax

import torch
import torch.nn.functional as F

@torch.no_grad()
def optimize_bb(grad_model, data, args,
                max_steps=200, gtol=0.5, alpha_init=1e-2,
                alpha_min=1e-6, alpha_max=1.0, dtype=torch.float32):

    data.to(device)     

    pos0, _, node_mask, _ = data_prepare(data=data, args=args)
    R = data['x_DBIM'].to(device)[node_mask.view(-1)].detach().clone().requires_grad_(False)

    init_pos_dif = F.l1_loss(sub_center(R), sub_center(pos0[node_mask.squeeze(-1)]))
    print('init pos diff:')
    print(round(init_pos_dif.item(), 12))

    initial_E, g  = energy_and_grad(data, node_mask, grad_model, R, args)
    initial_E_loss = F.l1_loss(initial_E, data.energy) + 1e-10
    force = -g                                        # 力
    # fmax = force.view(-1, 3).norm(dim=1).max().item()
    # initial_fmax = fmax 
    alpha = alpha_init

    prev_R, prev_g = None, None
    best_pos_dif = init_pos_dif
    start_patience = patience = 5
    best_E = initial_E

    batch_dic = data.batch.to(device)[node_mask.view(-1)].detach().clone()
    num_mols = int(batch_dic.max().item()) + 1

    for step in range(1, max_steps + 1):
        # --- Barzilai–Borwein 步长（需要上一轮的 R 和 梯度）

        if prev_R is None:
            alpha_m = torch.full((num_mols,), alpha_init, device=device, dtype=dtype)
        else: 
            s = R - prev_R                    # [N,3]
            y = g - prev_g                    # [N,3]

            ss_atom = (s * s).sum(dim=1)      # [N] 每个原子的 ||s||^2
            sy_atom = (s * y).sum(dim=1)      # [N] 每个原子的 s·y

            batch_eff = batch_dic

            SS = scatter_add(ss_atom, batch_eff, dim=0, dim_size=num_mols)  # [M]
            SY = scatter_add(sy_atom, batch_eff, dim=0, dim_size=num_mols)  # [M]

            alpha_m = SS / torch.clamp(SY, min=1e-15)                         # [M]
            alpha_m = alpha_m.clamp(min=alpha_min, max=alpha_max)

            # bad = SY.abs() < 1e-12
            # alpha_m[bad] = alpha_init

        alpha_b = alpha_m[batch_dic].unsqueeze(-1)           # [N,1]
        prev_R, prev_g = R.clone().detach(), g.clone().detach()
        R = (R + alpha_b * force).detach()               # 显式欧拉步 (F = -∇E)
        E, g = energy_and_grad(data, node_mask, grad_model, R, args)
        force = -g

        # if prev_R is not None:
        #     s = (R - prev_R)                       # ΔR
        #     y = (g - prev_g)                       # Δg
        #     sy = (s * y).sum()
        #     yy = (y * y).sum()
        #     if sy.abs() > 0:
        #         # 两种 BB 步长之一：α = (s·s)/(s·y) 或 α = (s·y)/(y·y)
        #         alpha = (s*s).sum() / sy.clamp(min=1e-15)
        #         alpha = float(alpha.clamp(min=alpha_min, max=alpha_max))

        # # --- 位置更新：R <- R + α * F（注意 F = -∇E）
        # prev_R = R.clone()
        # prev_g = g.clone()
        # R = (R + alpha * force).detach()               # 显式欧拉步
        # E, g = energy_and_grad(data, node_mask, grad_model, R, args)
        # force = -g



        # fmax = force.view(-1, 3).norm(dim=1).max().item()
        # pos_dif = F.l1_loss(sub_center(R), sub_center(pos0[node_mask.squeeze(-1)]))
        # if fmax < initial_fmax * gtol:
            # print('break')
            # break

        # print(fmax)

       
        # if E <= best_E:
        #     best_E = E
        #     patience = start_patience
        # else:
        #     patience -= 1

        # if patience == 0:
        #     print('early stop triggered')
        #     break

        # if step == max_steps:
        #     print(max_steps, ' steps finished')

    # pos_dif = F.l1_loss(sub_center(R), sub_center(pos0[node_mask.squeeze(-1)]))
    E_loss = F.l1_loss(E, data.energy)
    # print('final pos dif:')
    # print(round(pos_dif.item(), 12))
    # print('optimize dis:')
    # print(round((init_pos_dif - pos_dif).item(), 12))
    # print('fmax dis:')
    # print(initial_fmax - fmax)
    # print('E dis:')
    # print(initial_E.item() - E.item())
    # d_pos_dif = round((init_pos_dif - pos_dif).item(), 12)

    return 0, 0, (initial_E_loss - E_loss) / initial_E_loss

if __name__ == "__main__":

    args = parse_opt()

    device = args.device
    dtype = torch.float32

    model = PaiNN(use_pbc=False).to(device)
    model.load_state_dict(torch.load('/home/wzhao20/airp/saved_model/PaiNN-0819_e100-fft.pth')['model_state_dict'])

    train_loader, val_loader, test_loader = read_list("/scratch/wzhao20/data_rd/tmp", args, ext='difx')

    fail_c = 0
    unchange_c = 0
    cum_dis = 0
    cum_E = 0
    cum_E_loss_dis = 0

    for batch_idx, data in enumerate(val_loader):
        # print('molecule idx:', batch_idx)
        if batch_idx % 10 == 0:
            print(f"Batch [{batch_idx}/{len(val_loader)}] complete")

        pos_dif, E_dis, E_loss_dis = optimize_bb(grad_model=model, data=data, args=args)

        # print('opt dis:', pos_dif)
        cum_dis += pos_dif # expect > 0 
        cum_E += E_dis # expect > 0 
        cum_E_loss_dis += E_loss_dis # expect > 0 

        # pos_dif, E_dis = optimize_lbfgs_energy(model=model, data=data, args=args)

        if batch_idx >= 100:
            break

        # if pos_dif == 0:
        #     unchange_c += 1
        #     print('unchange case E dis:')
        #     print(E_dis)
        # elif pos_dif < 0:
        #     fail_c += 1
        #     print('fail case E dis:')
        #     print(E_dis)
    # print(cum_dis)
    # print(cum_E)
    print(cum_E_loss_dis)
    print(cum_E_loss_dis / 100)

    # print('average dis:', cum_dis / 20)

    # print("fail count:", fail_c)
    # print('unchange count:'. unchange_c)

    # print('val size:', len(val_loader))

