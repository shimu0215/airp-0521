
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
    return pred_PaiNN['energy'].to(device), -1*pred_PaiNN['forces'].to(device)     
    # return pred_PaiNN['energy'].to(device), data.energy_grad.to(device)[node_mask.view(-1)]  

@torch.no_grad()
def optimize_bb(grad_model, data, args,
                max_steps=50, gtol=0.5, alpha_init=1e-2,
                alpha_min=1e-6, alpha_max=1.0, dtype=torch.float32):

    data = data.to(device)     

    pos0, _, node_mask, _ = data_prepare(data=data, args=args)
    R = data['x_DBIM'].to(device)[node_mask.view(-1)].detach().clone().requires_grad_(False)
    # R = data['rdkit_pos'].to(device)[node_mask.view(-1)].detach().clone().requires_grad_(False)
    # R = pos0[0].to(device)[node_mask.view(-1)].detach().clone().requires_grad_(False)
    # sub_center(R)
    initial_E, g  = energy_and_grad(data, node_mask, grad_model, R, args)
    initial_E_loss = F.l1_loss(initial_E, data.energy) + 1e-15
    print('initial E loss:', initial_E_loss)
    force = -g                                        # 力

    alpha = alpha_init

    prev_R, prev_g = None, None

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

            bad = SY <= 0
            SY = torch.where(bad, torch.full_like(SY, 1.0), SY)

            alpha_m = SS / torch.clamp(SY, min=1e-15)                         # [M]
            alpha_m = alpha_m.clamp(min=alpha_min, max=alpha_max)
            alpha_m[bad] = alpha_init

            bad = SY.abs() < 1e-12
            alpha_m[bad] = alpha_init

        alpha_b = alpha_m[batch_dic].unsqueeze(-1) * 1e0           # [N,1]
        # alpha_b = 1e-2
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

    E_loss = F.l1_loss(E, data.energy.to(device))

    return 0, initial_E - E, (abs(initial_E_loss) - abs(E_loss)) / abs(initial_E_loss)

if __name__ == "__main__":

    args = parse_opt()

    device = args.device
    dtype = torch.float32

    model = PaiNN(use_pbc=False).to(device)
    model.load_state_dict(torch.load('/home/wzhao20/airp/saved_model/PaiNN-0819_e100-fft.pth')['model_state_dict'])

    train_loader, val_loader, test_loader = read_list("/scratch/wzhao20/data_rd/tmp", args, ext='difx')

    cum_E_loss_dis = 0
    E_loss_list = []

    pos_c = 0
    neg_c = 0

    pp = 0
    nn = 0
    pn = 0
    np = 0

    for batch_idx, data in enumerate(val_loader):
        # print('molecule idx:', batch_idx)
        if batch_idx % 10 == 0:
            print(f"Batch [{batch_idx}/{len(val_loader)}] complete")

        pos_dif, E_dis, E_loss_dis = optimize_bb(grad_model=model, data=data, args=args)

        cum_E_loss_dis += E_loss_dis # expect > 0 
        if E_loss_dis > 0: 
            pos_c += 1
            if E_dis > 0:
                pp += 1
            elif E_dis < 0:
                pn += 1
        elif E_loss_dis < 0:
            neg_c += 1
            if E_dis > 0:
                np += 1
            elif E_dis < 0:
                nn += 1

        E_loss_list.append(E_loss_dis.item())

        if batch_idx >= 1000:
            break


    print(cum_E_loss_dis / 100)
    print('pos:', pos_c)
    print('neg:', neg_c)

    import matplotlib.pyplot as plt

    save_path = '/home/wzhao20/DBIM_PaiNN/result-3'

    outliers = [x for x in E_loss_list if abs(x) > 1]
    print("Outliers (|x| > 1):", outliers)

    # 保留正常数据
    E_loss_list = [x for x in E_loss_list if abs(x) <= 1]

    plt.hist(E_loss_list, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of List')
    plt.show()

    plt.savefig(save_path, dpi=300)

    plt.close()  # 关闭图像窗口（防止多次绘图时重叠）   

    print(sum(E_loss_list) / len(E_loss_list))

    print(pp, pn, np, nn)