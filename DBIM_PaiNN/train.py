import math
from datetime import datetime
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from DBIM_argument import parse_opt
from DBIM_read_data import read_dataset
from utils import plot_result, read_list, load_model
from DBIM_models import DBIMGenerativeModel
from loss_functions import DBIMLoss, PaiNNLoss
from airp import make_noise_schedule, data_prepare, forward_transition, predict_PaiNN, predict_DBIM, sampling
from utils import sample_time_step, perturb_coordinates, sub_center
from eval import evaluate

from PaiNN.painn_model import PaiNN


def train(args, train_loader, val_loader):
    torch.autograd.set_detect_anomaly(True)
    device = args.device
    dtype = torch.float32

    generative_model = DBIMGenerativeModel(num_layers=args.num_layers, m_dim=args.m_dim).to(device)
    if args.load_DBIM:
        model_dict, optimizer_dict = load_model(model_path=args.DBIM_path, device=device, dtype=dtype, args=args)
        generative_model.load_state_dict(model_dict)
    # else:
    #     generative_model = DBIMGenerativeModel(num_layers=args.num_layers, m_dim=args.m_dim).to(device)

    pretrained_PaiNN = PaiNN(use_pbc=False).to(device)
    if args.load_PaiNN:
        pretrained_PaiNN.load_state_dict(torch.load(args.PaiNN_path)['model_state_dict'])

    for param in pretrained_PaiNN.parameters():
        param.requires_grad = False
    pretrained_PaiNN.eval()

    optimizer = torch.optim.AdamW(generative_model.parameters(), lr=args.lr, amsgrad=False, weight_decay=1e-12)
    if args.load_DBIM:
        optimizer.load_state_dict(optimizer_dict)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    print(optimizer.param_groups[0]['lr'])
    criterion = DBIMLoss()
    criterion_PaiNN = PaiNNLoss()

    T = args.T
    ats, bts, cts, rhos, sigmas = make_noise_schedule(T=T, eta=args.eta, device=device)

    epochs = args.epochs

    best_val_loss = float('inf')
    patience = args.patience
    counter = 0

    writer = SummaryWriter(log_dir=f"DBIM_log/whole_gamma_x0_D_type_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    batch_count = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            generative_model.train()
            optimizer.zero_grad()

            h, xt, xT, t_norm, node_mask, noise, x0, t = forward_transition(data=data, ats=ats, bts=bts, cts=cts, args=args)
            pos_predict, node_mask = predict_DBIM(h, xt, xT, t_norm, node_mask, generative_model, args)

            sigma_t = sigmas[t]
            loss_diff = criterion(model_predict=pos_predict, x0=x0, node_mask=node_mask, sigma_t=sigma_t, weighted=True)

            # loss, loss_energy, loss_force, loss_npa data, node_mask, pretrained_PaiNN, args, pos=None
            # pred_painn, data_painn = predict_PaiNN(data, node_mask, pretrained_PaiNN, args, pos=pos_predict)
            # loss_PaiNN = criterion_PaiNN(pred_painn, data_painn)
            # loss = loss_diff + loss_PaiNN[0]
            loss = loss_diff

            if not torch.isfinite(loss) or torch.isnan(loss):
                print("Skip step due to non-finite loss")
                optimizer.zero_grad(set_to_none=True)
                continue

            if any([torch.isnan(p.grad).any() for p in generative_model.parameters() if p.grad is not None]):
                optimizer.zero_grad(set_to_none=True)
                print('nan')
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(generative_model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss
            writer.add_scalar("train_loss/posDif_batch", loss, batch_count)
            batch_count += 1

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Train Loss: {loss.item():.10f}")

        with torch.no_grad():
            epoch_loss = epoch_loss / len(train_loader)
            writer.add_scalar("train_loss/posDif_epoch", epoch_loss, epoch)

            val_loss = evaluate(val_loader=val_loader, generative_model=generative_model, args=args)
            print(f"Epoch [{epoch + 1}/{epochs}] Val Loss: {val_loss:.10f}")
            writer.add_scalar("val_loss/posDif_epoch", val_loss, epoch)

            if args.eval_with_sampling and epoch % 3 == 0:
                posDif_sampling_val = 0.0
                propDif_sampling_val = torch.zeros(4).to(device)

                for batch_idx, data in enumerate(val_loader):
                    x_predict, x0_val, node_mask = sampling(data=data, ats=ats, bts=bts, cts=cts, rhos=rhos, generative_model=generative_model, args=args)
                    loss = F.mse_loss(x_predict * node_mask, x0_val * node_mask)
                    posDif_sampling_val += loss

                    if loss.item() <= 1:
                        pred_PaiNN, data_PaiNN = predict_PaiNN(data, node_mask, pretrained_PaiNN, args, pos=x_predict)
                        PaiNN_loss = criterion_PaiNN(pred=pred_PaiNN, data=data_PaiNN)
                        propDif_sampling_val += PaiNN_loss
                    else:
                        print('Fail when generate graph.')

                    if batch_idx % 50 == 0:
                        print(f"Epoch [{epoch + 1}] Batch [{batch_idx}/{len(val_loader)}] complete")

                print(f"Epoch [{epoch + 1}/{epochs}] Val Sampling Error: {posDif_sampling_val/len(val_loader):.10f}")
                writer.add_scalar("val_oss/sampledPosDif_epoch", posDif_sampling_val, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save({'model_state_dict': generative_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),}, args.DBIM_save_path)
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered")
                    break

    writer.close()


if __name__ == '__main__':
    args = parse_opt()
    train_loader, val_loader, test_loader = read_list(args.data_path, args)
    train(args, train_loader, val_loader)
