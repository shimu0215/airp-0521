import os
import re
import glob
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split


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

    original_batch = batch_transform(loaded_tensor[1]['z']) # atom
    natoms = nbatch_transform(loaded_tensor[1]['z'])

    data_list = []
    indices = loaded_tensor[1]['z']

    for i in range(n_molecules):
        start, end = indices[i], indices[i + 1]  # 获取当前分子范围

        data = Data(
        pos=database.pos[start:end], # atom
        atomic_numbers=database.z[start:end], # atom
        original_batch=original_batch[start:end], # atom
        natoms = natoms[i], # molecule
        energy = database.energy[i], # molecule
        npa = database.npa_charges[start:end], # atom
        species = torch.full((end-start,), species) # atom
    )
        data_list.append(data)

    # dataset = Data(
    #     pos=database.pos, # atom
    #     atomic_numbers=database.z, # atom
    #     original_batch=batch_transform(loaded_tensor[1]['z']), # atom
    #     natoms = nbatch_transform(loaded_tensor[1]['z']), # molecule
    #     energy = database.energy, # molecule
    #     npa = database.npa_charges, # atom
    #     species = torch.full((n,), species) # atom
    #
    #     # pos=database.pos[:37, :],
    #     # atomic_numbers=database.z[:37],
    #     # original_batch=batch_transform(loaded_tensor[1]['z'])[:37],
    #     # natoms=nbatch_transform(loaded_tensor[1]['z'])[:5],
    #     # y=database.y[:5]
    # )

    return data_list

def merge_dataset(data1, data2):

    offset = data1.original_batch.max() + 1
    data2.original_batch += offset

    merged_data = Batch.from_data_list(data1.to_data_list()+ [data2])

    return merged_data

def split_dataset(data, train_ratio=0.6, val_ratio=0.3):

    # molecule_indices = torch.arange(len(data.energy))
    # train_idx, temp_idx = train_test_split(data, train_size=train_ratio, random_state=42)

    train_set, temp_set = train_test_split(data, train_size=train_ratio, random_state=42)
    val_set, test_set = train_test_split(temp_set, train_size=val_ratio / (1 - train_ratio), random_state = 42)

    # val_idx, test_idx = train_test_split(temp_idx, train_size=val_ratio/(1-train_ratio), random_state=42)
    #
    # train_mask_molecule = torch.isin(molecule_indices, train_idx)
    # val_mask_molecule = torch.isin(molecule_indices, val_idx)
    # test_mask_molecule = torch.isin(molecule_indices, test_idx)
    #
    # train_mask_atom = torch.isin(data.original_batch, train_idx)
    # val_mask_atom = torch.isin(data.original_batch, val_idx)
    # test_mask_atom = torch.isin(data.original_batch, test_idx)
    #
    # data.train_mask_molecule = train_mask_molecule
    # data.val_mask_molecule = val_mask_molecule
    # data.test_mask_molecule = test_mask_molecule
    #
    # data.train_mask_atom = train_mask_atom
    # data.val_mask_atom = val_mask_atom
    # data.test_mask_atom = test_mask_atom



    return train_set, val_set, test_set


def read_dataset(path):

    file_list = glob.glob(os.path.join(path, "qm9star_*_chunk*_processed.pt"))

    pattern = re.compile(r"qm9star_(.+?)_chunk\d+_processed\.pt")

    data_list = []
    for file in file_list:
        match = pattern.search(file)
        name = match.group(1)

        loaded_tensor = torch.load(file)
        # loaded_tensor = torch.load('data/processed/qm9star_anion_chunk00_processed.pt')


        dataset = generate_a_sample_dataset(loaded_tensor, name)

        data_list.append(dataset)


    # data = data_list[0]

    train_list, val_list, test_list = split_dataset(data_list[0], train_ratio=0.6, val_ratio=0.3)

    train_loader = DataLoader(train_list, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=32)
    test_loader = DataLoader(test_list)

    # data = split_dataset(data, train_ratio=0.6, val_ratio=0.3)

    for batch in train_loader:

        print(batch)

    if len(data_list) > 1:
        data = Batch.from_data_list([data, data_list[1]])
        for i in range(2, len(data_list)):
            data = merge_dataset(data, data_list[i])

    data = split_dataset(data, train_ratio=0.6, val_ratio=0.3)

    d=DataLoader(data, batch_size=32, shuffle=True)

    # data = split_batch(data, batch_size=32)

    return data

if __name__ == "__main__":
    data_list = read_dataset("data/processed")




'''ole main'''

def nbatch_transform(indices):
    nbatch = torch.tensor([(indices[i + 1] - indices[i]).item() for i in range(len(indices) - 1)])

    return nbatch

def batch_transform(indices):
    batch = torch.tensor([i for i in range(len(indices) - 1) for _ in range(indices[i + 1] - indices[i])])

    return batch


def generate_a_sample_dataset():
    loaded_tensor = list(torch.load('data/processed/qm9star_anion_chunk00_processed.pt'))
    print(loaded_tensor)

    database = loaded_tensor[0]

    # Test in small
    dataset = Data(
        pos=database.pos[:37,:],
        atomic_numbers=database.z[:37],
        batch=batch_transform(loaded_tensor[1]['z'])[:37],
        natoms = nbatch_transform(loaded_tensor[1]['z'])[:5],
        y = database.y[:5]
    )

    return dataset


def generate_toy_dataset(num_samples: int = 100, num_atoms_range: tuple = (3, 10)):
    import numpy as np
    from torch_geometric.data import Data

    def random_molecule(num_atoms):
        """
        Generate a random molecule with the specified number of atoms.
        - x: Random node features.
        - pos: Random positions.
        - z: Random atomic numbers.
        - y: Random target property (e.g., energy).
        """
        x = torch.randn(num_atoms, 5)  # Random 5D node features
        pos = torch.randn(num_atoms, 3)  # Random 3D atom positions
        z = torch.randint(1, 10, (num_atoms,))  # Atomic numbers (H, C, O, etc.)
        y = torch.randn(1)  # Random scalar for molecular property
        return Data(x=x, pos=pos, atomic_numbers=z, y=y)

    dataset = [random_molecule(np.random.randint(*num_atoms_range)) for _ in range(num_samples)]
    return dataset

