import torch
import torch_cluster
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import Data
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet

from sklearn.model_selection import train_test_split

import pickle
import numpy as np
from datetime import datetime

from DBIM_argument import parse_opt
from PaiNN.painn_model import PaiNN
from DBIM_models import DBIMGenerativeModel
from loss_functions import ja4_Property_Predictive_Loss
from utils import read_list, load_model, get_hist
from airp import make_noise_schedule, forward_transition, predict_DBIM, predict_PaiNN, sampling
from loss_functions import DBIMLoss, PaiNNLoss
from airp import make_noise_schedule, data_prepare, forward_transition, predict_PaiNN, predict_DBIM, sampling
from utils import sample_time_step, perturb_coordinates, sub_center

from torch.utils.data import ConcatDataset



def read_data(data_path="/home/wzhao20/rp_data/ja4c_dataset/ja4c_data_large.pkl", train_ratio=0.5, val_ratio=0.1, batch_size=1, split=False):

    with open(data_path, "rb") as f:
        data_list = pickle.load(f)

    if split:
        train_set, temp_set = train_test_split(data_list, train_size=train_ratio, random_state=42)
        val_set, test_set = train_test_split(temp_set, train_size=val_ratio / (1 - train_ratio), random_state = 42)

    else:
        train_set = data_list
        val_set = data_list
        test_set = data_list

    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def train(data):
    generative_model.train()
    optimizer.zero_grad()

    h, xt, xT, t_norm, node_mask, noise, x0, t = forward_transition(data=data, ats=ats, bts=bts, cts=cts, args=args)
    pos_predict, node_mask = predict_DBIM(h, xt, xT, t_norm, node_mask, generative_model, args)

    sigma_t = sigmas[t]
    loss_diff = criterion(model_predict=pos_predict, x0=x0, node_mask=node_mask, sigma_t=sigma_t, weighted=True)

    loss = loss_diff

    return loss

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
g = torch.Generator()
g.manual_seed(seed)

args = parse_opt()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# load model
generative_model = DBIMGenerativeModel(num_layers=args.num_layers, m_dim=args.m_dim).to(device)
# generative_model_dict, _ = load_model(model_path=args.DBIM_path, device=device, dtype=dtype, args=args)
generative_model_dict, optimizer_dict = load_model(model_path='/home/wzhao20/DBIM_PaiNN/saved_model/DBIM-12-ja4c-128-highepo-1.pth', device=device, dtype=dtype, args=args)
generative_model.load_state_dict(generative_model_dict)

# load data
batch_size = 128
train_loader_regular, _, _ = read_data(data_path='/home/wzhao20/rp_data/ja4c_dataset/ja4c_data_regular.pkl', batch_size=batch_size, split=True)
train_loader_large, _, _ = read_data(data_path='/home/wzhao20/rp_data/ja4c_dataset/ja4c_data_large.pkl', batch_size=batch_size, split=False)

train_set = ConcatDataset([train_loader_regular.dataset, train_loader_large.dataset])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g)

# val_set = ConcatDataset([val_loader_regular.dataset, train_loader_large.dataset])
# val_loader = DataLoader(val_set, batch_size=8, shuffle=True)

# training setting
lr = 5e-4
optimizer = torch.optim.AdamW(generative_model.parameters(), lr=lr, amsgrad=False, weight_decay=1e-12)
optimizer.load_state_dict(optimizer_dict)
lr = optimizer.param_groups[0]['lr']
print('lr:', optimizer.param_groups[0]['lr'])

criterion = DBIMLoss()

T = args.T

ats, bts, cts, rhos, sigmas = make_noise_schedule(T=args.T, eta=args.eta, device=args.device)

best_loss = float('inf')
start_patience = patience = 3000
start_decay_patience = decay_patience = 1000

for epoch in range(5000000):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # print('training on:', batch_idx, ' in ', len(train_loader))
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

            loss_a = train(data_a)
            loss_c = train(data_c)
            loss_r = train(data_r)

            loss_batch = (loss_a + loss_c + loss_r) / 3

            if not torch.isfinite(loss_batch) or torch.isnan(loss_batch):
                print("Skip step due to non-finite loss")
                optimizer.zero_grad(set_to_none=True)
                continue

            if any([torch.isnan(p.grad).any() for p in generative_model.parameters() if p.grad is not None]):
                optimizer.zero_grad(set_to_none=True)
                print('nan')
                continue

            loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(generative_model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss_batch

        print('Epoch:', epoch, '; loss:', epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = start_patience
            decay_patience = start_decay_patience
            torch.save({'model_state_dict': generative_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),}, '/home/wzhao20/DBIM_PaiNN/saved_model/DBIM-12-ja4c-128-highepo-2.pth')
        else:
            patience -= 1
            decay_patience -= 1

        if decay_patience == 0:
            lr *= 0.5
            decay_patience = start_decay_patience
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            print('new lr: ', lr)

        if patience == 0:
            print('run out of patience - stop')
            break
