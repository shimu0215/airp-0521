# import net
import torch
import torch_cluster
import pickle

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
from loss_function import ja4_Loss, ja4_neutral_energy_ft_Loss
from eval import eval_model

from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split

def read_data(train_ratio=0.8, val_ratio=0.1, batch_size=1):
    # data_list = torch.load("/home/wzhao20/rp_data/ja4c_data.pkl")

    with open("/home/wzhao20/rp_data/ja4c_data.pkl", "rb") as f:
        data_list = pickle.load(f)

    # exclude = {
    #     'IUPAC_name', 'IUPAC_name_neat', 'txt_name', 'neat_txt_name', 'path',
    #     'class_type', 'charted', 'in_graph', 'si_geom', 'ja4_atom', 'ja4_charge'
    # }

    train_set, temp_set = train_test_split(data_list, train_size=train_ratio, random_state=42)
    val_set, test_set = train_test_split(temp_set, train_size=val_ratio / (1 - train_ratio), random_state = 42)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(data_list, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def freeze(mod):
    for p in mod.parameters():
        p.requires_grad = False

def unfreeze(mod):
    for p in mod.parameters():
        p.requires_grad = True

def eval_I_A(data):

    model.eval()
    with torch.no_grad():
        I_MAE = 0.0
        A_MAE = 0.0

        for idx, batch in enumerate(data):
            print('evaluating on:', idx, ' in ', len(data))
            batch = batch.to(device)

            data_a = batch.clone()
            data_c = batch.clone()
            data_r = batch.clone()

            data_a.species[:] = 0
            data_c.species[:] = 1
            data_r.species[:] = 3

            pred_a = model(data_a)
            pred_c = model(data_c)
            pred_r = model(data_r)

            I = batch.cation_scf - batch.radical_scf
            A = batch.radical_scf - batch.anion_scf

            I_pred = pred_c['energy'] - pred_r['energy']
            A_pred = pred_r['energy'] - pred_a['energy']

            error_A = F.l1_loss(A, A_pred)
            error_I = F.l1_loss(I, I_pred)

            I_MAE += error_I
            A_MAE += error_A

        I_MAE /= len(data)
        A_MAE /= len(data)

    return I_MAE, A_MAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PaiNN(use_pbc = False).to(device)
model.load_state_dict(torch.load('/home/wzhao20/airp/saved_model/PaiNN-0819-e100-ja4ft-all-4-MAE.pth')['model_state_dict'])

train_loader, val_loader, test_loader = read_data()

# freeze(model)
# unfreeze(model.out_energy)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
ja4_loss = ja4_neutral_energy_ft_Loss()

def train(data):
    model.train()
    optimizer.zero_grad()

    # data_c = data.clone()
    # data_a = data.clone()
    data_n = data.clone()

    # data_a.species[:] = 0
    # data_c.species[:] = 1
    data_n.species[:] = 2

    # pred_a = model(data_a)
    # pred_c = model(data_c)
    pred_n = model(data_n)

    # loss, loss_I, loss_A = ja4_loss(pred=[pred_a, pred_c, pred_n], data=data)
    loss = ja4_Loss(pred_n, data_n)

    loss.backward()
    optimizer.step()

    return loss, loss_I, loss_A

def eval(data):

    model.eval()
    with torch.no_grad():
        val_loss_a = 0.0
        val_loss_c = 0.0
        val_loss_r = 0.0

        c = 0
        for idx, batch in enumerate(data):
            print('evaluating on:', idx, ' in ', len(data))
            batch = batch.to(device)

            data_a = batch.clone()
            data_c = batch.clone()
            data_r = batch.clone()

            data_a.species[:] = 0
            data_c.species[:] = 1
            data_r.species[:] = 3

            pred_a = model(data_a)
            pred_c = model(data_c)
            pred_r = model(data_r)

            loss_a, loss_c, loss_r = ja4_loss(pred=[pred_a, pred_c, pred_r], data=batch)

            # if loss_a + loss_c + loss_r  > 10:
            #     # print(batch.natoms, batch[0].atomic_numbers.max())
            #     # print(loss_a, loss_c, loss_r)
            #     val_loss_a += 5e-2
            #     val_loss_c += 5e-2
            #     val_loss_r += 5e-2
            # else:
            val_loss_a += loss_a.item()
            val_loss_c += loss_c.item()
            val_loss_r += loss_r.item()
            # print(loss_a, loss_c, loss_r)

            if batch[0].atomic_numbers.max() > 9:
                c += 1

        val_loss_a /= len(data)
        val_loss_c /= len(data)
        val_loss_r /= len(data)

        print(c)

    return val_loss_a, val_loss_c, val_loss_r

# best_val_loss = np.inf
# start_patience = patience = 50

# for epoch in range(500):

#     num_batches = len(train_loader)
#     error_I = 0
#     error_A = 0
#     epoch_loss = 0

#     for i, batch in enumerate(train_loader):
#         batch = batch.to(device)
#         loss, loss_I, loss_A = train(batch)

#         error_A += loss_A
#         error_I += loss_I
#         epoch_loss += loss

#         if i % 5 == 0:
#             print(f"Epoch {epoch} batch {i}/{num_batches} : "
#                   f"total Loss = {loss.item():.4f} "
#                   f"loss_A = {loss_A.item():.4f} "
#                   f"loss_I = {loss_I.item():.3e} ")

#     print(f"Epoch {epoch} "
#           f"total_loss = {epoch_loss/num_batches:.4f}, "
#           f"error_A = {error_A/num_batches:.4f}, "
#           f"error_I = {error_I/num_batches:.3e}, ")

#     with torch.no_grad():
#         val_loss = eval(val_loader)[0]
#         print(f"Epoch {epoch}: val_Loss = {val_loss:.4f}")

#         if best_val_loss >= val_loss:
#             patience = start_patience
#             best_val_loss = val_loss
#             path = 'saved_model/'
#             torch.save({'model_state_dict': model.state_dict()}, path + 'saved_model/PaiNN-0806-2-ja4ft-neutral_energy_ft')
#         else:
#             patience -= 1

#         if patience <= 0:
#             print('Stopping training as validation accuracy did not improve '
#                 f'for {start_patience} epochs')
#             break

with torch.no_grad():

    # model.load_state_dict(torch.load('saved_model/PaiNN-0806-2-ja4ft-neutral_energy_ft')['model_state_dict'])
    test_loss = eval_I_A(test_loader)
    print('test_loss, test_loss_I, test_loss_A:', test_loss)
#     model.eval()
#     eval(test_loader=test_loader, model=model)
    # for batch in test_loader:
    #     test_data = batch.to(device)
    #     pred = model(test_data)
    #     loss, loss_energy, loss_force, loss_npa = loss_npa_energy_force(pred=pred, data=test_data)

    # print(f"total Loss = {loss:.4f} energy Loss = {loss_energy:.4f} forces Loss = {loss_force:.3e} npa_charges Loss = {loss_npa:.3e}")



