import torch
import torch_cluster
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset

from torch_geometric.data import Data
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet

from sklearn.model_selection import train_test_split

import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from DBIM_argument import parse_opt
from PaiNN.painn_model import PaiNN
from DBIM_models import DBIMGenerativeModel
from loss_functions import ja4_Property_Predictive_Loss
from utils import read_list, load_model, get_hist
from airp import make_noise_schedule, forward_transition, predict_DBIM, predict_PaiNN, sampling


def merge_dataloader(loader1, loader2):
    dataset1 = loader1.dataset
    dataset2 = loader2.dataset

    merged_dataset = ConcatDataset([dataset1, dataset2])

    merged_loader = DataLoader(
        merged_dataset,
        batch_size=loader1.batch_size,
        shuffle=True
    )

    return merged_loader

def read_data(data_path="/home/wzhao20/rp_data/ja4c_dataset/ja4c_data_large.pkl", train_ratio=0.8, val_ratio=0.1, batch_size=1, split=False):

    with open(data_path, "rb") as f:
        data_list = pickle.load(f)

    if split:
        train_set, temp_set = train_test_split(data_list, train_size=train_ratio, random_state=42)
        val_set, test_set = train_test_split(temp_set, train_size=val_ratio / (1 - train_ratio), random_state = 42)

    else:
        train_set = data_list
        val_set = data_list
        test_set = data_list

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader
    

def eval_property_prediction(data, metric='MAE'):

    x_predict, x0_val, node_mask = sampling(data=data, ats=ats, bts=bts, cts=cts, rhos=rhos,
                                            generative_model=generative_model, args=args, use_mol_type=True)
    pos_dif = F.l1_loss(x_predict * node_mask, x0_val * node_mask)

    energy_results = -1
    pred_PaiNN_ref, data_PaiNN_ref = predict_PaiNN(data, node_mask, pretrained_PaiNN, args, ja4=True)
    ref_energy_res = pred_PaiNN_ref['energy']

    if pos_dif.item() <= 1000000:
        pred_PaiNN, data_PaiNN = predict_PaiNN(data, node_mask, pretrained_PaiNN, args, pos=x_predict, ja4=True)
        
        if metric == 'MAE':
            loss = nn.L1Loss()
        else:
            loss = nn.MSELoss()

        energy_results = pred_PaiNN['energy']

        pred_ref_dis_energy = loss(energy_results, ref_energy_res)

        error_energy = loss(energy_results, data.energy)

        PaiNN_bias_energy = loss(ref_energy_res, data.energy)
    else:
        print('Fail when generate graph: ', pos_dif.item())
        return [0],0

    return energy_results, pred_ref_dis_energy, error_energy, PaiNN_bias_energy

def eval_I_A(data, ignore_bad_case=True):

    fail_gen = 0
    bad_case = 0
    pretrained_PaiNN.eval()
    generative_model.eval()
    with torch.no_grad():
        I_MAE = 0.0
        A_MAE = 0.0

        I_MAE_whole = 0.0
        A_MAE_whole = 0.0

        num_mol = 0
        num_mol_whole = 0

        for idx, batch in enumerate(data):
            # if idx % 100 == 0:
            #     print('evaluating on:', idx, ' in ', len(data))
            batch = batch.to(device)

            data_a = batch.clone()
            data_c = batch.clone()
            data_r = batch.clone()

            data_a.species[:] = 0
            data_c.species[:] = 1
            data_r.species[:] = 3

            data_a.energy = data_a.anion_scf
            data_c.energy = data_c.cation_scf
            data_r.energy = data_r.radical_scf

            # pred_a = pretrained_PaiNN(data_a)
            # pred_c = pretrained_PaiNN(data_c)
            # pred_r = pretrained_PaiNN(data_r)

            pred_a = eval_property_prediction(data_a)
            pred_c = eval_property_prediction(data_c)
            pred_r = eval_property_prediction(data_r)
            if pred_a[0][0] == 0 or pred_c[0][0] == 0 or pred_r[0][0] == 0:
                print(batch.natoms)
                fail_gen += len(batch)
                continue

            I = batch.cation_scf - batch.radical_scf
            A = batch.radical_scf - batch.anion_scf

            I_pred = pred_c[0] - pred_r[0]
            A_pred = pred_r[0] - pred_a[0]

            error_A = F.l1_loss(A, A_pred)
            error_I = F.l1_loss(I, I_pred)

            if ignore_bad_case:
                if error_I > 1 or error_A > 1:
                    print(error_A, error_I)
                    print(batch.natoms)
                    bad_case += len(batch)
                else:
                    I_MAE += error_I * len(batch)
                    A_MAE += error_A * len(batch)
                    num_mol += len(batch)

                I_MAE_whole += error_I * len(batch)
                A_MAE_whole += error_A * len(batch)
                num_mol_whole += len(batch)
            else:
                I_MAE += error_I * len(batch)
                A_MAE += error_A * len(batch)
                num_mol += len(batch)
            
            log_to_csv(csv_path, [batch[0].mol_id, 
            batch.anion_scf.item(), batch.cation_scf.item(), batch.radical_scf.item(), 
            pred_a[0].item(), pred_c[0].item(), pred_r[0].item()])

        I_MAE /= num_mol
        A_MAE /= num_mol
        I_MAE_whole /= num_mol_whole
        A_MAE_whole /= num_mol_whole

    return I_MAE, A_MAE, fail_gen, bad_case, I_MAE_whole, A_MAE_whole

def parameter_search(model, steps):

    generative_model_dict = torch.load(model, map_location=device)['model_state_dict']
    generative_model.load_state_dict(generative_model_dict)

    args.sample_steps = steps
    ignore_bad_case = True

    with torch.no_grad():

        test_loss = eval_I_A(test_loader, ignore_bad_case=ignore_bad_case)
        print('config setting: \n', 'Sampling Steps:', args.sample_steps, 
        '\n Ignore Bad Casees:', ignore_bad_case, 
        '\n Dataset: Whole Ja4c')
        # print('test_loss, test_loss_I, test_loss_A:', test_loss)
        print('Test_MAE_I:', test_loss[0].item())
        print('Test_MAE_A:', test_loss[1].item())
        return test_loss

import csv
import os

def init_csv(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", 
            "anion_scf_gt", "cation_scf_gt", "radical_scf_gt", 
            "anion_scf_pred", "cation_scf_pred", "radical_scf_pred"])

def log_to_csv(csv_path, res):
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(res)

csv_path = "log.csv"
init_csv(csv_path)

args = parse_opt()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

train_loader_regular, val_loader_regular, test_loader_regular = read_data(data_path='/home/wzhao20/rp_data/ja4c_dataset/ja4c_data_regular.pkl')
train_loader_large, val_loader_large, test_loader_large = read_data(data_path='/home/wzhao20/rp_data/ja4c_dataset/ja4c_data_large.pkl')
test_loader = merge_dataloader(test_loader_large, test_loader_regular)

ja4_loss = ja4_Property_Predictive_Loss()
ats, bts, cts, rhos, sigmas = make_noise_schedule(T=args.T, eta=args.eta, device=args.device)

pretrained_PaiNN = PaiNN(use_pbc = False).to(device)
pretrained_PaiNN.load_state_dict(torch.load('/home/wzhao20/airp/saved_model/PaiNN-0819-e100-ja4ft-all-4-MAE.pth')['model_state_dict'])

generative_model = DBIMGenerativeModel(num_layers=args.num_layers, m_dim=args.m_dim).to(device)

m1_path = '/home/wzhao20/DBIM_PaiNN/saved_model/DBIM-12-ja4c-128-highepo-1.pth'
m2_path = '/home/wzhao20/DBIM_PaiNN/saved_model/DBIM-12-ja4c-128-highepo-2.pth'
# steps_list = [10, 30, 50, 80, 100, 130, 150, 180, 200]
steps_list = [100]


m1_y1_list = []
m1_y2_list = []
m1_fail_gen = []
m1_bad_case = []
m1_y1_whole_list = []
m1_y2_whole_list = []
m2_y1_list = []
m2_y2_list = []
m2_fail_gen = []
m2_bad_case = []
m2_y1_whole_list = []
m2_y2_whole_list = []
for step in steps_list:
    y = parameter_search(m1_path, step)
    m1_y1_list.append(y[0].item())
    m1_y2_list.append(y[1].item())
    m1_fail_gen.append(y[2])
    m1_bad_case.append(y[3])
    m1_y1_whole_list.append(y[4].item())
    m1_y2_whole_list.append(y[5].item())

# for step in steps_list:
#     y = parameter_search(m2_path, step)
#     m2_y1_list.append(y[0].item())
#     m2_y2_list.append(y[1].item())
#     m2_fail_gen.append(y[2])
#     m2_bad_case.append(y[3])
#     m2_y1_whole_list.append(y[4].item())
#     m2_y2_whole_list.append(y[5].item())

# fig, ax1 = plt.subplots(figsize=(9, 5))


# l1 = ax1.plot(steps_list, m1_y1_list, marker="o", linestyle="-",
#               label="Model 1 - I")
# l2 = ax1.plot(steps_list, m1_y2_list, marker="o", linestyle="--",
#               label="Model 1 - A")
# l3 = ax1.plot(steps_list, m2_y1_list, marker="s", linestyle="-",
#               label="Model 2 - I")
# l4 = ax1.plot(steps_list, m2_y2_list, marker="s", linestyle="--",
#               label="Model 2 - A")

# ax1.set_xlabel("Hyperparameter")
# ax1.set_ylabel("Test Metric (I / A)")
# ax1.grid(True, linestyle="--", alpha=0.4)

# ax2 = ax1.twinx()

# l5 = ax2.plot(steps_list, m1_fail_gen, marker="^", linestyle=":",
#               label="Model 1 - Fail Gen")
# l6 = ax2.plot(steps_list, m1_bad_case, marker="v", linestyle=":",
#               label="Model 1 - Bad Case")
# l7 = ax2.plot(steps_list, m2_fail_gen, marker="^", linestyle="-.",
#               label="Model 2 - Fail Gen")
# l8 = ax2.plot(steps_list, m2_bad_case, marker="v", linestyle="-.",
#               label="Model 2 - Bad Case")

# ax2.set_ylabel("Failure / Bad Case")

# lines = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc="best", ncol=2)

# plt.title("Model Comparison across Hyperparameter")
# plt.tight_layout()

# plt.savefig(
#     "ja4c-parameter-search-dual-y-8-lines-200-500.png",
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.show()

# plt.figure(figsize=(9, 5))

# plt.plot(
#     steps_list, m1_y1_whole_list,
#     marker="o", linestyle="-",
#     label="Model 1 - I (Whole)"
# )
# plt.plot(
#     steps_list, m1_y2_whole_list,
#     marker="o", linestyle="--",
#     label="Model 1 - A (Whole)"
# )

# plt.plot(
#     steps_list, m2_y1_whole_list,
#     marker="s", linestyle="-",
#     label="Model 2 - I (Whole)"
# )
# plt.plot(
#     steps_list, m2_y2_whole_list,
#     marker="s", linestyle="--",
#     label="Model 2 - A (Whole)"
# )

# plt.xlabel("Hyperparameter")
# plt.ylabel("Test Metric (Whole)")
# plt.title("Model Comparison (Whole Set) across Hyperparameter")
# plt.grid(True, linestyle="--", alpha=0.4)
# plt.legend()
# plt.tight_layout()

# plt.savefig(
#     "ja4c-parameter-search-whole-200-500.png",
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.show()