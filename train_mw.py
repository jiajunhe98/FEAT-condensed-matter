import argparse
import torch
import numpy as np
from collections import defaultdict
from energy_mw import energy_mw, forces_mw, wrap_differences, wrap_coordinates

from networks.pegnn import *
from utils import Coef, remove_mean
from loss import v_loss
from copy import deepcopy
import time 
from tqdm import tqdm
import matplotlib.pyplot as plt

kj_to_kcal = 0.2390057361
BOLTZMANN_CONSTANT = 0.0019872067  # in units of kcal/mol K


def parse_args():
    parser = argparse.ArgumentParser(description='Training script with arguments')
    
    parser.add_argument('--phase', type=str, choices=['cubic', 'hexagonal'],
                        required=True, help='Phase type: cubic or hexagonal')
    
    parser.add_argument('--n_particles', type=int, choices=[64, 216, 512],
                        required=True, help='Number of particles: 64, 216, or 512')
    
    parser.add_argument('--budget', type=str, choices=['low', 'medium', 'high'],
                        required=True, help='Budget level: low, medium, or high')
    
    parser.add_argument('--seed', type=int, choices=[0, 1, 2],
                        required=True, help='Random seed: 0, 1, or 2')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(f"Phase: {args.phase}")
    print(f"Number of particles: {args.n_particles}")
    print(f"Budget: {args.budget}")
    print(f"Seed: {args.seed}")

    if args.phase == 'cubic':
        file_path = f'/data1/FEAT-JCP/data/mW/cubic/N{args.n_particles}/N{args.n_particles}-T200_001.npz'
    if args.phase == 'hexagonal':
        file_path = f'/data1/FEAT-JCP/data/mW/hexagonal/N{args.n_particles}/N{args.n_particles}-T200_001.npz'
    
    if args.budget == 'low':
        training_size = 10**3
    if args.budget == 'medium':
        training_size = 10**4
    if args.budget == 'high':
        training_size = 10**5

    # set torch and np seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # set default to float32
    torch.set_default_dtype(torch.float32)





    # -------------------------------
    # 2️⃣ Load one mW system file
    # -------------------------------
    mw_file = file_path

    data_mw = np.load(mw_file)

    positions_mw = torch.tensor(data_mw["pos"] * 10.0)       # nm -> Å
    boxes_mw_all = torch.diagonal(torch.tensor(data_mw["box"] * 10.0), dim1=1, dim2=2)
    boxes_mw = boxes_mw_all[0]  # only use the first box
    energies_mw_stored = torch.tensor(data_mw["ene"] * kj_to_kcal)
    forces_mw_stored = torch.tensor(data_mw["forces"] * kj_to_kcal / 10.0)
    equilibrium_lattice_mw = torch.tensor(data_mw["equi_lat"] * 10.0)

    # Energy and forces functions (first box)
    energy_fn_mw = lambda p: energy_mw(coordinates=p, box_length=boxes_mw.to(p.device))
    force_fn_mw  = lambda p: forces_mw(coordinates=p, box_length=boxes_mw.to(p.device))

    random_ids_mw = torch.randint(0, positions_mw.shape[0], (5,))
    energies_mw_recomputed = energy_fn_mw(positions_mw[random_ids_mw])
    forces_mw_recomputed = force_fn_mw(positions_mw[random_ids_mw])

    print("mW max energy diff:", torch.max(torch.abs(energies_mw_recomputed - energies_mw_stored[random_ids_mw])))
    print("mW max force diff:", torch.max(torch.abs(forces_mw_recomputed - forces_mw_stored[random_ids_mw])))

    # --------------


    box0_mw = boxes_mw
    box0 = box0_mw.float()

    wrap_displacement_fn = lambda p: wrap_differences(coordinate_deltas=p, box_length=box0.to(p.device))
    wrap_positions_fn = lambda p: wrap_coordinates(coordinate=p, box_length=box0.to(p.device), lower=0)

    training_idx = np.random.choice(positions_mw.shape[0], size=training_size, replace=False)
    displacements = wrap_displacement_fn(positions_mw[training_idx] - equilibrium_lattice_mw[None])

    # -------------------------------

    training_idx = np.random.choice(positions_mw.shape[0], size=training_size, replace=False)
    displacements = wrap_displacement_fn(positions_mw[training_idx] - equilibrium_lattice_mw[None])


    n_particles = args.n_particles
    beta = 1/(200*BOLTZMANN_CONSTANT)
    std = 0.2

    def get_spring_constant(beta, std):
        """Compute the optimal Einstein crystal spring constant given beta and std of displacements."""
        k_EC = 1.0 / (2.0 * beta * std**2)
        return k_EC
    k_EC = get_spring_constant(beta, std)

    def sample_displacements(n_samples):
        disp = std * torch.randn(n_samples, *(n_particles, 3))

        return disp - disp.mean(dim=1).unsqueeze(1)

    def energy_func_displacements(x):
        return torch.sum(k_EC * x**2, dim=(-2, -1))

    alpha = Coef('1-t')
    beta = Coef('t')
    device = 'cuda'
    egnn = PEGNN_dynamics(eq_pos=equilibrium_lattice_mw.float(),
                        box_edges=torch.diag(torch.tensor(data_mw['box'][0], dtype=torch.float32))*10,
                        rmin=0.2*10,
                        rmax=0.5*10,
                        box_edges_unit_cell=(torch.tensor([0.62, 0.62, 0.62])*10) if args.phase == 'cubic' else torch.tensor([4.3841, 7.5934, 7.1591]),
                        eq_pos_labeling_features=5,
                        n_features_t=10,
                        rbf_features=10,
                        rbf_trainable=False,
                        n_layers=10,
                        hidden_nf=32,
                        ).to(device)
    optimizer = torch.optim.Adam(egnn.parameters(), lr=1e-4)


    ema_net = deepcopy(egnn)


    def update_ema(model, model_ema, rate):
        for param, param_ema in zip(model.parameters(), model_ema.parameters()):
            param_ema.data = (1 - rate) * param.data + rate * param_ema.data


    vector_field = lambda t, x: egnn(t, x + equilibrium_lattice_mw[None].to(device).float().reshape(-1, n_particles* 3)) 
    LOSS = []

    train_iter = 1000000
    bsz = 128
    n_dim = 3

    for train_idx in tqdm(range(train_iter)):
        displacements_batch_idx = np.random.choice(training_size, size=bsz, replace=False)
        displacements_batch = displacements[displacements_batch_idx].reshape(bsz, -1).to(device).float()
        gaussian_noise = sample_displacements(bsz).reshape(bsz, -1).to(device).float()
        
        # # remove mean for both
        displacements_batch = remove_mean(displacements_batch, n_particles, n_dim, eq_lattice_com=0)
        gaussian_noise = remove_mean(gaussian_noise, n_particles, n_dim, eq_lattice_com=0)

        loss = v_loss(displacements_batch, gaussian_noise, alpha, beta, vector_field)

        optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(egnn.parameters(), max_norm=5.0)
        optimizer.step()
        
        LOSS.append(loss.item())
        if train_idx % 5000 == 0:
            print(f"Train iter {train_idx} loss: {loss.item()}")

        update_ema(egnn, ema_net, 0.999)

        if train_idx % 1000 == 0:
            # save ema network
            torch.save(ema_net.state_dict(), f"/data1/FEAT-JCP/network_ckpt/egnn_ema_phase{args.phase}_N{args.n_particles}_budget{args.budget}_seed{args.seed}.pth")

            
            plt.plot(LOSS[100:])
            plt.savefig(f"/data1/FEAT-JCP/network_ckpt/egnn_loss_phase{args.phase}_N{args.n_particles}_budget{args.budget}_seed{args.seed}.png")
            plt.close()
