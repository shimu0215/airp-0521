import os
import re
import glob
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


def read_list(path, args):
    file_list = glob.glob(os.path.join(path, "qm9star_*_chunk*_processed.pt"))

    data_list = []
    for file in file_list:

        loaded_tensor = torch.load(file, weights_only=False)

        data_list += loaded_tensor

    # train_ratio and val_ratio are the ratio of the entire dataset
    train_list, val_list, test_list = split_dataset(data_list, train_ratio=0.6, val_ratio=0.2)

    batch_size = 512

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size)
    test_loader = DataLoader(test_list, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def nbatch_transform(indices):
    nbatch = torch.tensor([(indices[i + 1] - indices[i]).item() for i in range(len(indices) - 1)])

    return nbatch

def batch_transform(indices):
    batch = torch.tensor([i for i in range(len(indices) - 1) for _ in range(indices[i + 1] - indices[i])])

    return batch


def generate_a_sample_dataset(loaded_tensor, species_name):

    database = loaded_tensor[0]

    n_atoms = len(database.z)
    n_molecules = len(database.y)

    species = -1
    if species_name == "anion":
        species = 0
    elif species_name == "cation":
        species = 1
    elif species_name == "neutral":
        species = 2
    elif species_name == "radical":
        species = 3

    pos = database.pos
    energy_grad = database.energy_grad
    z = database.z
    original_batch = batch_transform(loaded_tensor[1]['z'])
    natoms = nbatch_transform(loaded_tensor[1]['z'])
    energy = database.energy
    npa_charges = database.npa_charges

    data_list = []
    indices = loaded_tensor[1]['z']

    for i in range(n_molecules):
        start, end = indices[i], indices[i + 1]

        data = Data(
        pos=pos[start:end], # atom
        atomic_numbers=z[start:end], # atom
        # original_batch=original_batch[start:end], # atom
        batch=original_batch[start:end],  # atom
        natoms = natoms[i], # molecule
        energy = energy[i], # molecule
        energy_grad = energy_grad[start:end], # atom
        npa_charges = npa_charges[start:end], # atom
        species = torch.full((end-start,), species) # atom
    )
        data_list.append(data)

    return data_list

def split_dataset(data, train_ratio=0.6, val_ratio=0.3):

    train_set, temp_set = train_test_split(data, train_size=train_ratio, random_state=42)
    val_set, test_set = train_test_split(temp_set, train_size=val_ratio / (1 - train_ratio), random_state = 42)

    return train_set, val_set, test_set


def read_dataset(path):

    file_list = glob.glob(os.path.join(path, "qm9star_*_chunk*_processed.pt"))
    print(file_list)
    pattern = re.compile(r"qm9star_(.+?)_chunk\d+_processed\.pt")

    data_list = []
    for file in file_list:
        match = pattern.search(file)
        name = match.group(1)

        loaded_tensor = torch.load(file)
        dataset = generate_a_sample_dataset(loaded_tensor, name)

        data_list += dataset

    train_list, val_list, test_list = split_dataset(data_list, train_ratio=0.6, val_ratio=0.3)

    return train_list, val_list, test_list

if __name__ == "__main__":
    train_list, val_list, test_list = read_dataset("data/processed")

    train_loader = DataLoader(train_list, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_list, batch_size=32)
    test_loader = DataLoader(test_list)

    for batch_idx, batch in enumerate(test_loader):
        # batch = batch.to(device)  # 移动到 GPU（如果有）

        # 检查节点特征 x

        if batch.x is not None and (torch.isnan(batch.pos).any() or torch.isinf(batch.pos).any()):
            print(f"🚨 Found NaN/Inf in batch {batch_idx}: x")
        if batch.x is not None and (torch.isnan(batch.atomic_numbers).any() or torch.isinf(batch.atomic_numbers).any()):
            print(f"🚨 Found NaN/Inf in batch {batch_idx}: x")
        if batch.x is not None and (torch.isnan(batch.batch).any() or torch.isinf(batch.batch).any()):
            print(f"🚨 Found NaN/Inf in batch {batch_idx}: x")
        if batch.x is not None and (torch.isnan(batch.natoms).any() or torch.isinf(batch.natoms).any()):
            print(f"🚨 Found NaN/Inf in batch {batch_idx}: x")
        if batch.x is not None and (torch.isnan(batch.energy).any() or torch.isinf(batch.energy).any()):
            print(f"🚨 Found NaN/Inf in batch {batch_idx}: x")
        if batch.x is not None and (torch.isnan(batch.energy_grad).any() or torch.isinf(batch.energy_grad).any()):
            print(f"🚨 Found NaN/Inf in batch {batch_idx}: x")
        if batch.x is not None and (torch.isnan(batch.npa_charges).any() or torch.isinf(batch.npa_charges).any()):
            print(f"🚨 Found NaN/Inf in batch {batch_idx}: x")


    print(train_loader)