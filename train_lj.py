import argparse
import torch
import numpy as np
from collections import defaultdict
from energy_mw import wrap_differences, wrap_coordinates
from energy_lj import energy_lj, forces_lj 

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
    
    parser.add_argument('--phase', type=str, choices=['fcc', 'hcp'],
                        required=True, help='Phase type: fcc or hcp')
    
    parser.add_argument('--n_particles', type=int, choices=[108, 180, 256, 500],
                        required=True, help='Number of particles: 108, 180, 256, or 500')
    
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

    if args.phase == 'fcc':
        cut = {
            108: '2.00',
            180: '2.20',
            256: '2.70',
            500: '2.70',
        }
        file_path = f'/data1/FEAT-JCP/data/shift/fcc/N{args.n_particles}/N{args.n_particles}-T2.0000-RCUT{cut[args.n_particles]}_001.npz'
    if args.phase == 'hcp':
        file_path = f'/data1/FEAT-JCP/data/shift/hcp/N{args.n_particles}/N{args.n_particles}-T2.0000-RCUT2.20_001.npz'
    
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
    # 1️⃣ Load one LJ system file
    # -------------------------------
    lj_file = file_path 

    data_lj = np.load(lj_file)

    positions_lj = torch.tensor(data_lj["pos"] * 10.0)      # nm -> Å
    boxes_lj_all = torch.diagonal(torch.tensor(data_lj["box"] * 10.0), dim1=1, dim2=2)
    energies_lj_stored = torch.tensor(data_lj["ene"] * kj_to_kcal)
    forces_lj_stored = torch.tensor(data_lj["forces"] * kj_to_kcal / 10.0)
    equilibrium_lattice_lj = torch.tensor(data_lj["equi_lat"] * 10.0)

    # Define energy and forces functions (first box)
    boxes_lj = boxes_lj_all[0]  # only use the first box
    if args.phase == 'fcc':
        rcut = float(cut[args.n_particles])  # RCUT from filename
    if args.phase == 'hcp':
        rcut = 2.2
    energy_fn_lj = lambda p: energy_lj(coordinates=p, box_length=boxes_lj, cutoff=rcut)
    force_fn_lj  = lambda p: forces_lj(coordinates=p, box_length=boxes_lj, cutoff=rcut)

    # Sample random subset for checking
    random_ids = torch.randint(0, positions_lj.shape[0], (5,))
    energies_lj_recomputed = energy_fn_lj(positions_lj[random_ids])
    forces_lj_recomputed = force_fn_lj(positions_lj[random_ids])

    print("LJ max energy diff:", torch.max(torch.abs(energies_lj_recomputed - energies_lj_stored[random_ids])))
    print("LJ max force diff:", torch.max(torch.abs(forces_lj_recomputed - forces_lj_stored[random_ids])))



    box0 = boxes_lj.float()
    wrap_displacement_fn = lambda p: wrap_differences(coordinate_deltas=p, box_length=box0.to(p.device))
    wrap_positions_fn = lambda p: wrap_coordinates(coordinate=p, box_length=box0.to(p.device), lower=0)


    wrap_displacement_fn = lambda p: wrap_differences(coordinate_deltas=p, box_length=box0.to(p.device))
    wrap_positions_fn = lambda p: wrap_coordinates(coordinate=p, box_length=box0.to(p.device), lower=0)

    training_idx = np.random.choice(positions_lj.shape[0], size=training_size, replace=False)
    displacements = wrap_displacement_fn(positions_lj[training_idx] - equilibrium_lattice_lj[None])


    n_particles = args.n_particles
    T = 239.5324979454125 
    beta = 1/(T*BOLTZMANN_CONSTANT)
    sigma = 3.4
    std = 0.05 * sigma




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
    if args.phase == 'fcc':
        if args.n_particles in [108, 256, 500]:
            box_edges_unit_cell = torch.tensor([4.9708, 4.9708, 4.9708])
        if args.n_particles == 180:
            box_edges_unit_cell = torch.tensor([3.5149, 6.0880, 8.6097])
    if args.phase == 'hcp':
        box_edges_unit_cell = torch.tensor([3.5149, 6.0880, 5.7398])
    egnn = PEGNN_dynamics(eq_pos=equilibrium_lattice_lj.float(),
                        box_edges=torch.diag(torch.tensor(data_lj['box'][0], dtype=torch.float32))*10,
                        rmin=0.27*10,
                        rmax=0.45*10,
                        box_edges_unit_cell=box_edges_unit_cell,
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


    vector_field = lambda t, x: egnn(t, x + equilibrium_lattice_lj[None].to(device).float().reshape(-1, n_particles* 3)) 
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
            torch.save(ema_net.state_dict(), f"/data1/FEAT-JCP/network_lj_ckpt/egnn_ema_phase{args.phase}_N{args.n_particles}_budget{args.budget}_seed{args.seed}.pth")

            
            plt.plot(LOSS[100:])
            plt.savefig(f"/data1/FEAT-JCP/network_lj_ckpt/egnn_loss_phase{args.phase}_N{args.n_particles}_budget{args.budget}_seed{args.seed}.png")
            plt.close()
