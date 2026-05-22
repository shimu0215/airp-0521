import torch
import torch.nn as nn
import torch.nn.functional as F

class ja4_Loss(nn.Module):
    def __init__(self):
        super(ja4_Loss, self).__init__()

    def forward(self, pred, data):
        pred_a, pred_c, pred_n = pred[0], pred[1], pred[2]

        I = data.cation_energy - data.neutral_energy
        A = data.neutral_energy - data.anion_energy

        I_pred = pred_c['energy'] - pred_n['energy']
        A_pred = pred_n['energy'] - pred_a['energy']

        error_A = abs(A - A_pred) / abs(A)
        error_I = abs(I - I_pred) / abs(I)

        mae_loss = nn.L1Loss()
        dis_c = mae_loss(data.cation_energy, pred_c['energy'])
        dis_a = mae_loss(data.anion_energy, pred_a['energy'])
        dis_n = mae_loss(data.neutral_energy, pred_n['energy'])

        total_loss = (5 * error_A + 0.0 * error_I).mean()
        print(dis_a, dis_c, dis_n)
        # print(I.mean(), A.mean())
        # total_loss = (dis_c + dis_a + dis_n) / 3

        return total_loss, error_I.mean(), error_A.mean()

class ja4_neutral_energy_ft_Loss(nn.Module):
    def __init__(self):
        super(ja4_neutral_energy_ft_Loss, self).__init__()

    def forward(self, pred, data, metric='MAE'):
        pred_n = pred

        if metric == 'MAE':
            loss = nn.L1Loss()
        else:
            loss = nn.MSELoss()
        # mae_loss = nn.L1Loss()
        # dis_n = loss(data.energy, pred_n['energy'])
        dis_a = loss(data.anion_scf, pred_n[0]['energy'])
        dis_c = loss(data.cation_scf, pred_n[1]['energy'])
        dis_r = loss(data.radical_scf, pred_n[2]['energy'])
        # dis_n = mae_loss(data.neutral_energy, pred_n['energy'])

        # print(data.radical_scf - pred_n[0]['energy'])

        return dis_a, dis_c, dis_r


class energy_force_npa_Loss(nn.Module):
    def __init__(self):
        super(energy_force_npa_Loss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, data):
        energy = pred['energy']
        energy_grad = pred['forces']
        npa_charges = pred['npa_charges']

        # loss_energy = F.mse_loss(energy, data.energy)
        # loss_force = F.mse_loss(energy_grad, data.energy_grad)
        # loss_npa = F.mse_loss(npa_charges, data.npa_charges)

        loss_energy = F.l1_loss(energy, data.energy)
        loss_force = F.l1_loss(energy_grad, data.energy_grad)
        loss_npa = F.l1_loss(npa_charges, data.npa_charges)

        # lambda_1 = 1.0 / (loss_energy.item() + 1e-6)
        # lambda_2 = 1.0 / (loss_force.item() + 1e-6)
        # lambda_3 = 1.0 / (loss_npa.item() + 1e-6)

        # lambda_1 = 0.05
        # lambda_2 = 0.75 * 10**10
        # lambda_3 = 0.2
        lambda_1 = 1
        lambda_2 = 1 * 10**2
        lambda_3 = 20

        total_loss = lambda_1 * loss_energy + lambda_2 * loss_force + lambda_3 * loss_npa

        # total_loss = torch.log(1 + loss_energy) + torch.log(1 + loss_force) + torch.log(1 + loss_npa)

        return total_loss, loss_energy, loss_force, loss_npa

class energy_force_npa_mse_Loss(nn.Module):
    def __init__(self):
        super(energy_force_npa_mse_Loss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, data):
        energy = pred['energy']
        energy_grad = pred['forces']
        npa_charges = pred['npa_charges']

        loss_energy = F.mse_loss(energy, data.energy)
        loss_force = F.mse_loss(energy_grad, data.energy_grad)
        loss_npa = F.mse_loss(npa_charges, data.npa_charges)

        lambda_1 = 1
        lambda_2 = 1 * 10**4
        lambda_3 = 20

        total_loss = lambda_1 * loss_energy + lambda_2 * loss_force + lambda_3 * loss_npa

        return total_loss, loss_energy, loss_force, loss_npa

class energy_force_npa_e100_Loss(nn.Module):
    def __init__(self):
        super(energy_force_npa_e100_Loss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, data):
        energy = pred['energy']
        energy_grad = pred['forces']
        npa_charges = pred['npa_charges']

        loss_energy = F.l1_loss(energy, data.energy)
        loss_force = F.l1_loss(energy_grad, data.energy_grad)
        loss_npa = F.l1_loss(npa_charges, data.npa_charges)

        lambda_1 = 100
        lambda_2 = 1 * 10**(4 + 3)
        lambda_3 = 20

        total_loss = lambda_1 * loss_energy + lambda_2 * loss_force + lambda_3 * loss_npa

        return total_loss, loss_energy, loss_force, loss_npa

class energy_force_npa_eval(nn.Module):
    def __init__(self, metric):
        super(energy_force_npa_eval, self).__init__()
        if metric == "mae":
            self.measure = nn.L1Loss()
        elif metric =='mse':
            self.measure = nn.MSELoss()

    def forward(self, pred, data):
        result = {}

        energy = pred['energy']
        energy_grad = pred['forces']
        npa_charges = pred['npa_charges']

        loss_energy = self.measure(energy, data.energy)
        loss_force = self.measure(energy_grad, data.energy_grad)
        loss_npa = self.measure(npa_charges, data.npa_charges)

        # loss_energy /= len(data)
        # loss_force /= len(data)
        # loss_npa /= len(data)

        result['energy'] = loss_energy
        result['force'] = loss_force
        result['npa'] = loss_npa

        return result