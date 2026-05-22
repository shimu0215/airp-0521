import torch
import torch_cluster
from datetime import datetime
from painn import PaiNN
import torch.nn.functional as F
from torch_geometric.data import Data

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet

from preprocess import read_dataset, read_list
from loss_function import energy_force_npa_Loss
from eval import eval_model

from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_list, val_list, test_list = read_dataset("processed-real")

train_loader = DataLoader(train_list, batch_size=64, shuffle=True)
val_loader = DataLoader(val_list, batch_size=512)

checkpoint = torch.load('saved_model/PaiNN-0903-rob-9-5.pth')
model = PaiNN(use_pbc = False).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']

print('lr:', lr)

loss_npa_energy_force = energy_force_npa_Loss()

writer = SummaryWriter(log_dir=f"log/0903_rob_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def add_noise(data):

    n_atom_tmp = data.pos.size(0)
    noise_ratio = 10
    n_atom_noise = n_atom_tmp // noise_ratio
    noise_idx = torch.randperm(n_atom_tmp)[:n_atom_noise]

    noise_std = 0.01
    noise = torch.randn((n_atom_noise, 3), device=device) * noise_std

    pos_new = data.pos.clone()
    pos_new[noise_idx] = pos_new[noise_idx] + noise

    data = data.clone()
    data.pos = pos_new

    return data

def train(data):
    data = add_noise(data)
    model.train()
    optimizer.zero_grad()

    pred = model(data)

    loss, loss_energy, loss_force, loss_npa = loss_npa_energy_force(pred=pred, data=data)

    if any([torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None]):
        optimizer.zero_grad(set_to_none=True); 
        return NaN, NaN, NaN, NaN

    loss.backward()
    optimizer.step()

    return loss, loss_energy, loss_force, loss_npa

best_val_loss = np.inf
start_patience = patience = 30
start_decay_patience = decay_patience = 5

# best_val_loss = checkpoint['best_val_loss']
# patience = checkpoint['patience']
# decay_patience = checkpoint['decay_patience']

batch_count = 0

for epoch in range(500):

    # for batch in train_loader:
    num_batches = len(train_loader)
    epoch_loss = 0.0
    epoch_loss_energy = 0.0
    epoch_loss_force = 0.0
    epoch_loss_npa = 0.0

    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        loss, loss_energy, loss_force, loss_npa = train(batch)
        if torch.isnan(loss):
            continue

        if not torch.isfinite(loss) or torch.isnan(loss):
            print("Skip step due to non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            continue

        if any([torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None]):
            optimizer.zero_grad(set_to_none=True)
            print('nan')
            continue

        epoch_loss += loss.item()
        epoch_loss_energy += loss_energy.item()
        epoch_loss_force += loss_force.item()
        epoch_loss_npa += loss_npa.item()

        writer.add_scalar("train_loss/train_total_loss", loss.item(), batch_count+1)
        writer.add_scalar("train_loss/train_energy_loss", loss_energy.item(), batch_count+1)
        writer.add_scalar("train_loss/train_force_loss", loss_force.item(), batch_count+1)
        writer.add_scalar("train_loss/train_npa_loss", loss_npa.item(), batch_count+1)

        if i % 100 == 0:
            print(f"Epoch {epoch} batch {i}/{num_batches} : "
                  f"total Loss = {loss:.4f} "
                  f"energy Loss = {loss_energy:.4f} "
                  f"forces Loss = {loss_force:.3e} "
                  f"npa_charges Loss = {loss_npa:.3e}")

        batch_count += 1

    print(f"Epoch {epoch} "
          f"total_loss = {epoch_loss/num_batches:.4f}, "
          f"energy_loss = {epoch_loss_energy/num_batches:.4f}, "
          f"forces_loss = {epoch_loss_force/num_batches:.3e}, "
          f"npa_charges_loss = {epoch_loss_npa/num_batches:.3e} ")

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_energy_loss = 0.0
        val_force_loss = 0.0
        val_npa_loss = 0.0
        for batch in val_loader:  # Validation dataset
            batch = batch.to(device)

            batch = add_noise(batch)

            pred = model(batch)
            loss, loss_energy, loss_force, loss_npa = loss_npa_energy_force(pred=pred, data=batch)
            val_loss += loss.item()
            val_energy_loss += loss_energy.item()
            val_force_loss += loss_force.item()
            val_npa_loss += loss_npa.item()

        val_loss /= len(val_loader)
        val_energy_loss /= len(val_loader)
        val_force_loss /= len(val_loader)
        val_npa_loss /= len(val_loader)

        writer.add_scalar("val_loss/val_total_loss", val_loss, epoch)
        writer.add_scalar("val_loss/val_energy_loss", val_energy_loss, epoch)
        writer.add_scalar("val_loss/val_force_loss", val_force_loss, epoch)
        writer.add_scalar("val_loss/val_npa_loss", val_npa_loss, epoch)


        print(f"Epoch {epoch}: val_Loss = {val_loss:.4f}")

        if best_val_loss >= val_loss:
            patience = start_patience
            decay_patience = start_decay_patience
            best_val_loss = val_loss
            path = 'saved_model/'
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'patience': patience,
                        'decay_patience': decay_patience,
                        'best_val_loss': best_val_loss}, path + 'PaiNN-0903-rob-9-6.pth')
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

