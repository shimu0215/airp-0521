# import net
import torch
import torch_cluster
# import torch_scatter
# import torch_sparse
# import torch_spline_conv

from datetime import datetime

# from fairchem.core.models.painn import PaiNN
from painn import PaiNN
import torch.nn.functional as F
from torch_geometric.data import Data

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet

from preprocess import read_dataset
# from preprocess import read_dataset
from loss_function import ja4_Loss, ja4_neutral_energy_ft_Loss, energy_force_npa_Loss
from eval import eval_model

from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split

def read_data(train_ratio=0.8, val_ratio=0.1, batch_size=32):
    data_list = torch.load("../rp_data/dataset.pt")
    # train_set, val_set, test_set = read_dataset("processed")

    exclude = {
        'IUPAC_name', 'IUPAC_name_neat', 'txt_name', 'neat_txt_name', 'path',
        'class_type', 'charted', 'in_graph', 'si_geom', 'ja4_atom', 'ja4_charge'
    }

    train_set, temp_set = train_test_split(data_list, train_size=train_ratio, random_state=42)
    val_set, test_set = train_test_split(temp_set, train_size=val_ratio / (1 - train_ratio), random_state = 42)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, exclude_keys=exclude)
    val_loader = DataLoader(val_set, batch_size=batch_size, exclude_keys=exclude)
    test_loader = DataLoader(test_set, batch_size=batch_size, exclude_keys=exclude)

    return train_loader, val_loader, test_loader

def freeze(mod):
    for p in mod.parameters():
        p.requires_grad = False

def unfreeze(mod):
    for p in mod.parameters():
        p.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PaiNN(use_pbc = False).to(device)
model.load_state_dict(torch.load('saved_model/PaiNN-0819_e100.pth')['model_state_dict'])

train_loader, val_loader, test_loader = read_data()

# freeze(model)
# unfreeze(model.out_energy)

lr=1e-3
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
ja4_loss = ja4_neutral_energy_ft_Loss()

def train(data):
    model.train()
    optimizer.zero_grad()

    # data_c = data.clone()
    # data_a = data.clone()
    # data_n = data.clone()

    # data_a.species[:] = 2
    # data_c.species[:] = 1
    data.species[:] = 2
    # print(data.species)

    # pred_a = model(data_a)
    # pred_c = model(data_c)
    pred = model(data)

    # loss, loss_I, loss_A = ja4_loss(pred=[pred_a, pred_c, pred_n], data=data)
    loss = ja4_loss(pred, data, metric="MAE")
    # print(loss)

    loss.backward()
    optimizer.step()

    return loss

def eval(data):

    model.eval()
    with torch.no_grad():
        val_loss = 0.0

        for batch in data:
            batch = batch.to(device)

            batch.species[:] = 2

            pred_n = model(batch)
            loss = ja4_loss(pred_n, batch, metric="MAE")

            val_loss += loss.item()

        val_loss /= len(data)

    return val_loss

best_val_loss = np.inf
start_patience = patience = 1000
decay_patience = start_decay_patience = 200

for epoch in range(10000):

    num_batches = len(train_loader)
    epoch_loss = 0

    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        loss = train(batch)

        epoch_loss += loss

        # if i % 5 == 0:
        #     print(f"Epoch {epoch} batch {i}/{num_batches} : "
        #           f"total Loss = {loss.item():.4f} ")

    print(f"Epoch {epoch} "
          f"total_loss = {epoch_loss/num_batches:.4f}, ")

    with torch.no_grad():
        val_loss = eval(val_loader)
        print(f"Epoch : val_Loss = {val_loss:.4f}")

        if best_val_loss >= val_loss:
            patience = start_patience
            decay_patience = start_decay_patience
            best_val_loss = val_loss
            path = 'saved_model/'
            torch.save({'model_state_dict': model.state_dict()}, path + 'PaiNN-0819_e100-ja4ft-neutral_energy_ft-2-whole')
        else:
            patience -= 1
            decay_patience -= 1

        if decay_patience <= 0:
            lr *= 0.5
            decay_patience = start_decay_patience
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            print('new lr: ')
            print(lr)

        if patience <= 0:
            print('Stopping training as validation accuracy did not improve '
                f'for {start_patience} epochs')
            break

with torch.no_grad():

    model.load_state_dict(torch.load('saved_model/PaiNN-0819_e100-ja4ft-neutral_energy_ft-2')['model_state_dict'])
    # model.load_state_dict(torch.load('saved_model/PaiNN-0819_e100.pth')['model_state_dict'])
    test_loss = eval(test_loader)
    print('test_loss:', test_loss)
