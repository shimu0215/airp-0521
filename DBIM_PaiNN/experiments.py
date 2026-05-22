import torch
from torch_geometric.loader import DataLoader
from PaiNN.painn_model import PaiNN
import torch.nn.functional as F
from airp import make_noise_schedule, forward_transition, predict_DBIM, predict_PaiNN, sampling
import torch
import torch.nn.functional as F
import torch.nn as nn

from loss_functions import DBIMLoss, PaiNNLoss
from airp import make_noise_schedule, forward_transition, predict_DBIM, predict_PaiNN, sampling
from DBIM_argument import parse_opt
from utils import read_list, load_model, get_hist
from PaiNN.painn_model import PaiNN
from DBIM_models import DBIMGenerativeModel

@torch.no_grad()
def eval_ja4():
    # ats, bts, cts, rhos, sigmas = make_noise_schedule(T=args.T, eta=args.eta, device=args.device)
    # criterion_PaiNN = PaiNNLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1

    data_list = torch.load("../rp_data/dataset_rd_2.pt")

    exclude = {
        'IUPAC_name', 'IUPAC_name_neat', 'txt_name', 'neat_txt_name', 'path',
        'class_type', 'charted', 'in_graph', 'si_geom', 'ja4_atom', 'ja4_charge'
    }
    dataset = DataLoader(data_list, batch_size=batch_size, shuffle=False, exclude_keys=exclude)

    pretrained_PaiNN = PaiNN(use_pbc=False).to(device)
    # pretrained_PaiNN.load_state_dict(torch.load('saved_model/PaiNN-0806-2')['model_state_dict'])
    pretrained_PaiNN.load_state_dict(torch.load('saved_model/PaiNN-0806-2-ja4ft')['model_state_dict'])
    pretrained_PaiNN.eval()

    # error_I_minus_A = 0
    # error_I_plus_A = 0

    error_I = 0
    error_A = 0
    error_n_energy = 0

    fail_number = 0

    for batch_idx, data in enumerate(dataset):
        data.to(device)

        if not hasattr(data, 'id'):
            if hasattr(data, 'batch'):
                num_graphs = int(data.batch.max().item()) + 1
                data.id = torch.arange(num_graphs, device=data.batch.device, dtype=torch.long)
                data.sid = torch.arange(num_graphs, device=data.batch.device, dtype=torch.long)
                data.fid = torch.arange(num_graphs, device=data.batch.device, dtype=torch.long)
            else:
                data.id = torch.tensor([0], device=data.x.device, dtype=torch.long)

        # data_c = data.clone()
        # data_a = data.clone()
        data_n = data.clone()

        # data_a.species[:] = 0
        # data_c.species[:] = 1
        data_n.species[:] = 2

        try:
            # pred_a = pretrained_PaiNN(data_a)
            # pred_c = pretrained_PaiNN(data_c)
            pred_n = pretrained_PaiNN(data_n)
            print(pred_n['energy'] - data.neutral_energy)

            error_n_energy += F.l1_loss(pred_n['energy'], data.neutral_energy)

            # I = data.cation_energy - data.neutral_energy
            # A = data.neutral_energy - data.anion_energy

            # I_pred = pred_c['energy'] - pred_n['energy']
            # A_pred = pred_n['energy'] - pred_a['energy']

            # error_A += abs(A - A_pred) / abs(A)
            # error_I += abs(I - I_pred) / abs(I)

            # I_minus_A = data.cation_energy - data.anion_energy
            # I_plus_A = data.anion_energy + data.cation_energy

            # I_minus_A_pred = pred_c['energy'] - pred_a['energy']
            # I_plus_A_pred = pred_a['energy'] + pred_c['energy']

            # print(I_minus_A)
            # print(I_minus_A_pred)

            # error_I_minus_A += abs(I_minus_A - I_minus_A_pred) / abs(I_minus_A)
            # error_I_plus_A += abs(I_plus_A - I_plus_A_pred) / abs(I_plus_A)
            if batch_idx % 10 ==0:
                print(f"Batch [{batch_idx}/{len(dataset)}] complete")
        except Exception as e:
            print(f"Batch [{batch_idx}/{len(dataset)}] fail")
            print(data_list[batch_idx].txt_name)
            fail_number += 1

    # print('error of I - A:', error_I_minus_A / (len(dataset) - fail_number))
    # print('error of I + A:', error_I_plus_A / (len(dataset) - fail_number))

    # print('error of I:', error_I / (len(dataset) - fail_number))
    # print('error of A:', error_A / (len(dataset) - fail_number))
    print('error of neutral energy:', error_n_energy / (len(dataset) - fail_number))

    return 0

def genrerate_pos_ja4c(dataset, generative_model, pretrained_PaiNN, args):
    with torch.no_grad():
        device = args.device

        generative_model.eval()
        pretrained_PaiNN.eval()

        ats, bts, cts, rhos, sigmas = make_noise_schedule(T=args.T, eta=args.eta, device=args.device)

        accumulate_pos_dif = 0

        accumulate_A_error = 0
        accumulate_I_error = 0

        for batch_idx, data in enumerate(dataset):
            # print(batch_idx)
            data.to(device)
            data.smiles = 'C'

            data_a = data.clone()
            data_a.atomic_numbers_one_hot = data_a.atomic_numbers_one_hot_a
            data_a.rdkit_pos = data_a.pos_a
            data_a.species[:] = 0
            data_a.atomic_numbers = data_a.atomic_numbers_a
            data_a.natoms = data_a.natoms_a
            data_a.atom_mask = data_a.atom_mask_a

            data_c = data.clone()
            data_c.atomic_numbers_one_hot = data_c.atomic_numbers_one_hot_c
            data_c.rdkit_pos = data_c.pos_c
            data_c.species[:] = 1
            data_c.atomic_numbers = data_c.atomic_numbers_c
            data_c.natoms = data_c.natoms_c
            data_c.atom_mask = data_c.atom_mask_c

            data_n = data.clone()
            data_n.atomic_numbers_one_hot = data_n.atomic_numbers_one_hot_n
            data_n.rdkit_pos = data_n.pos_n
            data_n.species[:] = 2
            data_n.atomic_numbers = data_n.atomic_numbers_n
            data_n.natoms = data_n.natoms_n
            data_n.atom_mask = data_n.atom_mask_n

            data_r = data.clone()
            data_r.atomic_numbers_one_hot = data_r.atomic_numbers_one_hot_r
            data_r.rdkit_pos = data_r.pos_r
            data_r.species[:] = 3
            data_r.atomic_numbers = data_r.atomic_numbers_r
            data_r.natoms = data_r.natoms_r
            data_r.atom_mask = data_r.atom_mask_r

            x_predict_a, x0_val_a, node_mask_a = sampling(data=data_a, ats=ats, bts=bts, cts=cts, rhos=rhos,
                                                    generative_model=generative_model, args=args, use_mol_type=True)

            pred_PaiNN_a, data_PaiNN_a = predict_PaiNN(data_a, node_mask_a, pretrained_PaiNN, args, pos=x_predict_a, ja4=True)
            # accumulate_pos_dif += F.l1_loss(x_predict_a * node_mask_a, x0_val_a * node_mask_a)
            

            x_predict_c, x0_val_c, node_mask_c = sampling(data=data_c, ats=ats, bts=bts, cts=cts, rhos=rhos,
                                                    generative_model=generative_model, args=args, use_mol_type=True)
            pred_PaiNN_c, data_PaiNN_c = predict_PaiNN(data_c, node_mask_c, pretrained_PaiNN, args, pos=x_predict_c, ja4=True)

            # accumulate_pos_dif += F.l1_loss(x_predict_c * node_mask_c, x0_val_c * node_mask_c)


            x_predict_r, x0_val_r, node_mask_r = sampling(data=data_r, ats=ats, bts=bts, cts=cts, rhos=rhos,
                                                    generative_model=generative_model, args=args, use_mol_type=True)
            pred_PaiNN_r, data_PaiNN_r = predict_PaiNN(data_r, node_mask_r, pretrained_PaiNN, args, pos=x_predict_r, ja4=True)
            # accumulate_pos_dif += F.l1_loss(x_predict_r * node_mask_r, x0_val_r * node_mask_r)

            x_predict_n, x0_val_n, node_mask_n = sampling(data=data_n, ats=ats, bts=bts, cts=cts, rhos=rhos,
                                                    generative_model=generative_model, args=args, use_mol_type=True)
            pred_PaiNN_n, data_PaiNN_n = predict_PaiNN(data_n, node_mask_n, pretrained_PaiNN, args, pos=x_predict_n, ja4=True)
            # accumulate_pos_dif += F.l1_loss(x_predict_n * node_mask_n, x0_val_n * node_mask_n)

            I_pred = pred_PaiNN_c['energy'] - pred_PaiNN_r['energy']
            A_pred = pred_PaiNN_r['energy'] - pred_PaiNN_a['energy']

            I = data.cation_energy - data.neutral_energy
            A = data.neutral_energy - data.anion_energy

            I_diff = torch.abs(I_pred - I)
            A_diff = torch.abs(A_pred - A)

            accumulate_I_error += I_diff
            accumulate_A_error += A_diff

        #     if pos_dif.item() <= 1:
        #         pred_PaiNN, data_PaiNN = predict_PaiNN(data, node_mask, pretrained_PaiNN, args, pos=x_predict)
        #         pred_PaiNN_ref, data_PaiNN_ref = predict_PaiNN(data, node_mask, pretrained_PaiNN, args)
        #         # print(pred_PaiNN_ref)

        #         diff_energy = torch.abs(pred_PaiNN['energy'] - pred_PaiNN_ref['energy']).view(-1)
        #         diff_force = torch.abs(pred_PaiNN['forces'] - pred_PaiNN_ref['forces']).view(-1)
        #         diff_npa = torch.abs(pred_PaiNN['npa_charges'] - pred_PaiNN_ref['npa_charges']).view(-1)

        #         energy_diff.append(diff_energy)
        #         force_diff.append(diff_force)
        #         npa_diff.append(diff_npa)

        #         if metric == 'MAE':
        #             loss = nn.L1Loss()
        #         else:
        #             loss = nn.MSELoss()

        #         energy_dis = loss(pred_PaiNN['energy'], pred_PaiNN_ref['energy'])
        #         force_dis = loss(pred_PaiNN['forces'], pred_PaiNN_ref['forces'])
        #         npa_dis = loss(pred_PaiNN['npa_charges'], pred_PaiNN_ref['npa_charges'])
        #         dis = torch.stack([energy_dis, force_dis, npa_dis], dim=0)
        #         diffusion_bias += dis

        #         curr_error = criterion_PaiNN(pred=pred_PaiNN, data=data_PaiNN)
        #         # print(curr_error)
        #         # curr_error = criterion_PaiNN(pred=pred_PaiNN_ref, data=data_PaiNN_ref)
        #         error += curr_error
        #         ref += criterion_PaiNN(pred=pred_PaiNN_ref, data=data_PaiNN_ref)
        #         # energy_diff.append(curr_error[1].item())
        #     else:
        #         print('Fail when generate graph: ', pos_dif.item())

        #     if batch_idx % 10 == 0:
        #         print(f"Batch [{batch_idx}/{len(val_loader)}] complete")
        
        # # get_hist(diff_list=energy_diff, path='plots/energy_')

        # diffusion_bias = diffusion_bias / len(val_loader)
        # error = error / len(val_loader)

        # print('Metric:', metric)
        # print('Error:', 'Energy -', error[1].item(), 'force -', error[2].item(), 'npa -', error[3].item())
        # print('Error by Diffusion:', 'Energy -', diffusion_bias[0].item(), 'force -', diffusion_bias[1].item(), 'npa -', diffusion_bias[2].item())
        # print(ref / len(val_loader))

        # print(accumulate_pos_dif/4/len(data))
        print('A:', accumulate_A_error/len(data))
        print('I:', accumulate_I_error/len(data))


if __name__ == "__main__":
    args = parse_opt()

    device = args.device
    dtype = torch.float32

    generative_model = DBIMGenerativeModel(num_layers=args.num_layers, m_dim=args.m_dim, dim=21).to(device)
    if args.load_DBIM:
        model_dict, optimizer_dict = load_model(model_path='/home/wzhao20/DBIM_PaiNN/saved_model/DBIM_model_gamma_type_4_highlr-16-6.pth', device=device, dtype=dtype, args=args)
        generative_model.load_state_dict(model_dict)

    pretrained_PaiNN = PaiNN(use_pbc=False).to(device)
    if args.load_PaiNN:
        pretrained_PaiNN.load_state_dict(torch.load('/home/wzhao20/DBIM_PaiNN/saved_model/PaiNN-0819_e100.pth')['model_state_dict'])

    batch_size = 1

    data_list = torch.load("/home/wzhao20/rp_data/dataset_rd_3.pt")

    exclude = {
        'IUPAC_name', 'IUPAC_name_neat', 'txt_name', 'neat_txt_name', 'path',
        'class_type', 'charted', 'in_graph', 'si_geom', 'ja4_atom', 'ja4_charge'
    }
    dataset = DataLoader(data_list, batch_size=batch_size, shuffle=False, exclude_keys=exclude)

    with torch.no_grad():
        genrerate_pos_ja4c(dataset, generative_model, pretrained_PaiNN, args)
    