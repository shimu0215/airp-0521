import glob
import os
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch_geometric.loader import DataLoader

from DBIM_read_data import read_dataset, split_dataset
from DBIM_models import DBIMGenerativeModel
from DBIM_argument import parse_opt

def read_list(path, args, ext=None):
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