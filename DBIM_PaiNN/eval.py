import torch
import torch.nn.functional as F
import torch.nn as nn
import glob
import os

from loss_functions import DBIMLoss, PaiNNLoss
from airp import make_noise_schedule, forward_transition, predict_DBIM, predict_PaiNN, sampling
from DBIM_argument import parse_opt
from utils import read_list, load_model, get_hist
from PaiNN.painn_model import PaiNN
from DBIM_models import DBIMGenerativeModel
from torch_geometric.loader import DataLoader


def evaluate(val_loader, generative_model, args):
    with torch.no_grad():
        generative_model.eval()

        DBIM_criterion = DBIMLoss()

        ats, bts, cts, rhos, sigmas = make_noise_schedule(T=args.T, eta=args.eta, device=args.device)

        eval_loss = 0.0
        for batch_idx, data in enumerate(val_loader):
            h, xt, xT, t_norm, node_mask, noise, x0, t = forward_transition(data=data, ats=ats, bts=bts, cts=cts,
                                                                            args=args)
            pos_predict, node_mask = predict_DBIM(h, xt, xT, t_norm, node_mask, generative_model, args)
            loss = DBIM_criterion(model_predict=pos_predict,  x0=x0, node_mask=node_mask)

            eval_loss += loss

        eval_loss = eval_loss / len(val_loader)

        return eval_loss

def eval_pos_generation(val_loader, generative_model, args):
    with torch.no_grad():
        generative_model.eval()

        ats, bts, cts, rhos, sigmas = make_noise_schedule(T=args.T, eta=args.eta, device=args.device)

        accumulate_pos_dif = 0

        for batch_idx, data in enumerate(val_loader):
            x_predict, x0_val, node_mask = sampling(data=data, ats=ats, bts=bts, cts=cts, rhos=rhos,
                                                    generative_model=generative_model, args=args, use_mol_type=True)
            pos_dif = F.l1_loss(x_predict[node_mask.squeeze(-1)], x0_val[node_mask.squeeze(-1)])
            if pos_dif.item() <= 1:
                accumulate_pos_dif += pos_dif
            else:
                print('Fail when generate graph: ', pos_dif.item())

            if batch_idx % 10 == 0:
                print(f"Batch [{batch_idx}/{len(val_loader)}] complete")
        
        print("pos dif:", accumulate_pos_dif / len(val_loader))

        return pos_dif

def eval_property_prediction(val_loader, generative_model, pretrained_PaiNN, args, metric, plot=True):
    with torch.no_grad():
        generative_model.eval()
        pretrained_PaiNN.eval()

        ats, bts, cts, rhos, sigmas = make_noise_schedule(T=args.T, eta=args.eta, device=args.device)
        criterion_PaiNN = PaiNNLoss()

        diffusion_bias = torch.zeros(3).to(args.device)
        error = torch.zeros(4).to(args.device)
        ref = torch.zeros(4).to(args.device)

        energy_diff = []
        force_diff = []
        npa_diff = []

        for batch_idx, data in enumerate(val_loader):
            x_predict, x0_val, node_mask = sampling(data=data, ats=ats, bts=bts, cts=cts, rhos=rhos,
                                                    generative_model=generative_model, args=args, use_mol_type=True)
            pos_dif = F.l1_loss(x_predict * node_mask, x0_val * node_mask)

            if pos_dif.item() <= 1:
                pred_PaiNN, data_PaiNN = predict_PaiNN(data, node_mask, pretrained_PaiNN, args, pos=x_predict)
                pred_PaiNN_ref, data_PaiNN_ref = predict_PaiNN(data, node_mask, pretrained_PaiNN, args)
                # print(pred_PaiNN_ref)

                diff_energy = torch.abs(pred_PaiNN['energy'] - pred_PaiNN_ref['energy']).view(-1)
                diff_force = torch.abs(pred_PaiNN['forces'] - pred_PaiNN_ref['forces']).view(-1)
                diff_npa = torch.abs(pred_PaiNN['npa_charges'] - pred_PaiNN_ref['npa_charges']).view(-1)

                energy_diff.append(diff_energy)
                force_diff.append(diff_force)
                npa_diff.append(diff_npa)

                if metric == 'MAE':
                    loss = nn.L1Loss()
                else:
                    loss = nn.MSELoss()

                energy_dis = loss(pred_PaiNN['energy'], pred_PaiNN_ref['energy'])
                force_dis = loss(pred_PaiNN['forces'], pred_PaiNN_ref['forces'])
                npa_dis = loss(pred_PaiNN['npa_charges'], pred_PaiNN_ref['npa_charges'])
                dis = torch.stack([energy_dis, force_dis, npa_dis], dim=0)
                diffusion_bias += dis

                curr_error = criterion_PaiNN(pred=pred_PaiNN, data=data_PaiNN)
                # print(curr_error)
                # curr_error = criterion_PaiNN(pred=pred_PaiNN_ref, data=data_PaiNN_ref)
                error += curr_error
                ref += criterion_PaiNN(pred=pred_PaiNN_ref, data=data_PaiNN_ref)
                # energy_diff.append(curr_error[1].item())
            else:
                print('Fail when generate graph: ', pos_dif.item())

            if batch_idx % 10 == 0:
                print(f"Batch [{batch_idx}/{len(val_loader)}] complete")
        
        # get_hist(diff_list=energy_diff, path='plots/energy_')

        diffusion_bias = diffusion_bias / len(val_loader)
        error = error / len(val_loader)

        print('Metric:', metric)
        print('Error:', 'Energy -', error[1].item(), 'force -', error[2].item(), 'npa -', error[3].item())
        print('Error by Diffusion:', 'Energy -', diffusion_bias[0].item(), 'force -', diffusion_bias[1].item(), 'npa -', diffusion_bias[2].item())
        print(ref / len(val_loader))

        # get_hist(diff_list=force_diff, path='plots/force_')
        # get_hist(diff_list=npa_diff, path='plots/npa_')
        
        # with open('plots/energy_n.txt','w') as f:
        #     for item in energy_diff:
        #         f.write(f"{item}\n")

        return diffusion_bias, error

def save_generated_pos(path, generative_model, args):
    file_list = glob.glob(os.path.join(path, "qm9star_*_chunk*_processed.pt"))

    batch_size = args.batch_size
    ats, bts, cts, rhos, sigmas = make_noise_schedule(T=args.T, eta=args.eta, device=args.device)

    generative_model.eval()
    with torch.no_grad():
        print('file number: ', len(file_list))
        for file in file_list:
            print(f"\nProcessing: {file}")
            new_dataset = []

            loaded_tensor = torch.load(file, weights_only=False)
            data_loader = DataLoader(loaded_tensor, batch_size=batch_size, shuffle=False)

            for batch_idx, data in enumerate(data_loader):
                fail_generate = False
                x_predict, x0_val, node_mask = sampling(data=data, ats=ats, bts=bts, cts=cts, rhos=rhos,
                                                        generative_model=generative_model, args=args, use_mol_type=True)
                pos_dif = F.l1_loss(x_predict[node_mask.squeeze(-1)], x0_val[node_mask.squeeze(-1)])
                if pos_dif.item() >= 1:
                    print('Fail when generate graph: ', pos_dif.item())
                    fail_generate = True
                
                data_list = data.to_data_list()
                for i, d in enumerate(data_list):
                    d.x_DBIM = x_predict[i].detach().cpu()  
                    new_dataset.append(d)
                    

                if batch_idx % 10 == 0:
                    print(f"Batch [{batch_idx}/{len(data_loader)}] complete")

            base, ext = os.path.splitext(file)
            save_path = f"{base}_difx{ext}"
            torch.save(new_dataset, save_path)
            print(f"Saved with predictions → {save_path}")



if __name__ == "__main__":
    args = parse_opt()

    device = args.device
    dtype = torch.float32

    generative_model = DBIMGenerativeModel(num_layers=args.num_layers, m_dim=args.m_dim).to(device)
    if args.load_DBIM:
        model_dict, optimizer_dict = load_model(model_path=args.DBIM_path, device=device, dtype=dtype, args=args)
        generative_model.load_state_dict(model_dict)

    # save_generated_pos(args.data_path, generative_model, args)

    pretrained_PaiNN = PaiNN(use_pbc=False).to(device)
    if args.load_PaiNN:
        pretrained_PaiNN.load_state_dict(torch.load(args.PaiNN_path)['model_state_dict'])

    train_loader, val_loader, test_loader = read_list(args.data_path, args)

    with torch.no_grad():
        eval_property_prediction(test_loader, generative_model, pretrained_PaiNN, args, metric='MAE')
        # print(evaluate(test_loader, generative_model, args))
        # eval_pos_generation(val_loader, generative_model, args)