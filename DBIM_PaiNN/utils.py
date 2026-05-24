import glob
import os
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from DBIM_read_data import read_dataset, split_dataset
from DBIM_models import DBIMGenerativeModel
from DBIM_argument import parse_opt
from packed_rd import PACKED_RD_INDEX, PackedRDChunkDataset


def _split_cache_path(path, train_ratio, val_ratio, seed):
    filename = f"split_indices_t{train_ratio:.4f}_v{val_ratio:.4f}_s{seed}.pt"
    return os.path.join(path, filename)


def _load_or_create_split_indices(path, dataset_size, train_ratio, val_ratio, seed=42):
    cache_path = _split_cache_path(path, train_ratio, val_ratio, seed)
    if os.path.exists(cache_path):
        saved = torch.load(cache_path, map_location="cpu")
        if saved.get("dataset_size") == dataset_size:
            return saved["train"], saved["val"], saved["test"]

    rng = np.random.default_rng(seed)
    indices = rng.permutation(dataset_size)

    train_end = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    val_end = train_end + val_size

    train_idx = indices[:train_end].tolist()
    val_idx = indices[train_end:val_end].tolist()
    test_idx = indices[val_end:].tolist()

    torch.save(
        {
            "dataset_size": dataset_size,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "seed": seed,
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        },
        cache_path,
    )
    return train_idx, val_idx, test_idx


def _build_loaders_from_packed(path, args):
    dataset = PackedRDChunkDataset(
        path,
        cache_size=getattr(args, "packed_cache_size", 2),
    )
    seed = getattr(args, "split_seed", 42)
    train_idx, val_idx, test_idx = _load_or_create_split_indices(
        path,
        len(dataset),
        args.train_ratio,
        args.val_ratio,
        seed=seed,
    )

    batch_size = args.batch_size
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size)
    return train_loader, val_loader, test_loader

def read_list(path, args, ext=None):
    packed_index_path = os.path.join(path, PACKED_RD_INDEX)
    if os.path.exists(packed_index_path):
        return _build_loaders_from_packed(path, args)

    if ext is None:
        file_list = glob.glob(os.path.join(path, "qm9star_*_chunk*_processed.pt"))
    else:
        file_list = glob.glob(os.path.join(path, "qm9star_*_chunk*_processed_difx.pt"))

    data_list = []
    for file in file_list:

        loaded_tensor = torch.load(file, weights_only=False)

        data_list += loaded_tensor

    # train_ratio and val_ratio are the ratio of the entire dataset
    train_list, val_list, test_list = split_dataset(data_list, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    batch_size = args.batch_size

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size)
    test_loader = DataLoader(test_list, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def read_dataloader(args):

    batch_size = args.batch_size

    train_list, val_list, test_list = read_dataset(
        args.data_path,
        max_atom_number=args.max_atom_number, max_atom_id = args.max_atom_id,
        train=args.train_ratio, val=args.val_ratio)

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size)
    test_loader = DataLoader(test_list, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def plot_result(result_list, name):
    x = list(range(len(result_list))) 
    y = result_list

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label='Result')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('Result Trend')
    plt.grid(True)
    plt.legend()

    plt.savefig('vis/'+name, dpi=300, bbox_inches='tight') 

def load_DBIM_without_type(model_path, device, dtype, args):
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    return state_dict, 0


def load_model(model_path, device, dtype, args):
    ckpt = torch.load(model_path, map_location=device)

    return ckpt['model_state_dict'], ckpt['optimizer_state_dict']

def sample_time_step(x, T):
    t_val = torch.randint(
        low=0,
        high=T,
        size=(x.size(0), 1, 1),
        device=x.device
    )
    t_val = t_val.expand(-1, x.size(1), -1)

    return t_val


def perturb_coordinates(x0, noise_std=0.5):

    noise = torch.randn_like(x0) * noise_std
    xT = x0 + noise
    return xT


def sub_center(x):

    center_of_mass = x.mean(dim=-2, keepdim=True)
    result = x - center_of_mass
    return result

def get_hist(diff_list, path):
    diff_tensor = torch.cat(diff_list).cpu().numpy()

    min_val = diff_tensor.min()
    max_val = diff_tensor.max()
    bins = np.linspace(min_val, max_val, num=51)
    
    plt.hist(diff_tensor, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Absolute Difference')
    plt.ylabel('Frequency')
    plt.title('Distribution of Absolute Differences')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path + 'abs_diff_hist.png', dpi=300)
    plt.close()
