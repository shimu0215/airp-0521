import torch
from torch_geometric.loader import DataLoader

from painn import PaiNN
from preprocess import read_dataset
from loss_function import energy_force_npa_eval


def eval_model(test_loader, model):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval() 

    n = len(test_loader)
    print('batch number:', n)


    evaluator_mae = energy_force_npa_eval(metric='mae')
    evaluator_mse = energy_force_npa_eval(metric='mse')

    mae_energy = 0.0
    mae_force = 0.0
    mae_npa = 0.0

    mse_energy = 0.0
    mse_force = 0.0
    mse_npa = 0.0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            test_data = batch.to(device)
            if i == 0:
                print('batch size:', len(test_data))

            pred = model(test_data)

            result_mae = evaluator_mae(pred=pred, data=test_data)
            result_mse = evaluator_mse(pred=pred, data=test_data)

            mae_energy += result_mae['energy']
            mae_force += result_mae['force']
            mae_npa += result_mae['npa']

            mse_energy += result_mse['energy']
            mse_force += result_mse['force']
            mse_npa += result_mse['npa']

    print(f"Model evaluate: \n --Metric: MAE-- \n" 
    f"energy = {mae_energy/n:.3e} forces = {mae_force/n:.3e} npa_charges = {mae_npa/n:.3e}"
    f"\n --Metric: MSE-- \n"
    f"energy = {mse_energy/n:.3e} forces = {mse_force/n:.3e} npa_charges = {mse_npa/n:.3e}")

def read_data(path, cutoff=12.0):
    # _, _, test_list = read_dataset("../data_rd/data_rd_test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_list = read_dataset("processed-real")
    test_loader = DataLoader(test_list, batch_size=512)

    model = PaiNN(use_pbc = False, cutoff=cutoff).to(device)

    record = torch.load(path)
    model.load_state_dict(record['model_state_dict'])

    model = model.to(device)

    return test_loader, model


if __name__ == "__main__":
    test_loader, modle = read_data(path = "saved_model/PaiNN-0819_e100.pth", cutoff=5.0)
    eval_model(test_loader=test_loader, model=modle)
