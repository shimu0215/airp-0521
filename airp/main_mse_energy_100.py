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
from loss_function import energy_force_npa_e100_Loss
from eval import eval_model

from torch.utils.tensorboard import SummaryWriter

def freeze(mod):
    for p in mod.parameters():
        p.requires_grad = False

def unfreeze(mod):
    for p in mod.parameters():
        p.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('saved_model/PaiNN-0819_e100-fft.pth')

train_list, val_list, test_list = read_dataset("processed-real")

train_loader = DataLoader(train_list, batch_size=64, shuffle=True)
val_loader = DataLoader(val_list, batch_size=512)
test_loader = DataLoader(test_list)

model = PaiNN(use_pbc = False).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

freeze(model)
unfreeze(model.out_forces)

lr = 1e-4
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']

print('lr:')
print(lr)

loss_npa_energy_force = energy_force_npa_e100_Loss()

writer = SummaryWriter(log_dir=f"log/0819_e100_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def train(data):
    model.train()
    optimizer.zero_grad()

    pred = model(data)

    loss, loss_energy, loss_force, loss_npa = loss_npa_energy_force(pred=pred, data=data)

    loss.backward()
    optimizer.step()

    return loss, loss_energy, loss_force, loss_npa

best_val_loss = np.inf
start_patience = patience = 30
start_decay_patience = decay_patience = 5

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
                        'optimizer_state_dict': optimizer.state_dict(),}, path + 'PaiNN-0819_e100-fft.pth')
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

# with torch.no_grad():
#     model.eval()
#     eval(test_loader=test_loader, model=model)
    # for batch in test_loader:
    #     test_data = batch.to(device)
    #     pred = model(test_data)
    #     loss, loss_energy, loss_force, loss_npa = loss_npa_energy_force(pred=pred, data=test_data)

    # print(f"total Loss = {loss:.4f} energy Loss = {loss_energy:.4f} forces Loss = {loss_force:.3e} npa_charges Loss = {loss_npa:.3e}")



