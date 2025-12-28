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
    
    parser.add_argument('--test_budget', type=str, choices=['low', 'medium', 'high'],
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
        file_path = f'/data1/FEAT-JCP/data/shift/fcc/N{args.n_particles}/N{args.n_particles}-T2.0000-RCUT{cut[args.n_particles]}_002.npz'
    if args.phase == 'hcp':
        file_path = f'/data1/FEAT-JCP/data/shift/hcp/N{args.n_particles}/N{args.n_particles}-T2.0000-RCUT2.20_002.npz'
    
    if args.test_budget == 'low':
        training_size = 10**2
    if args.test_budget == 'medium':
        training_size = 10**3
    if args.test_budget == 'high':
        training_size = 10**4

    # set torch and np seed
    torch.manual_seed(args.seed*10)
    np.random.seed(args.seed*10)
    # set default to float32
    torch.set_default_dtype(torch.float32)

    print(f"Test size: {training_size}")



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
    energy_fn_lj = lambda p: energy_lj(coordinates=p, box_length=boxes_lj.to(p.device), cutoff=rcut)
    force_fn_lj  = lambda p: forces_lj(coordinates=p, box_length=boxes_lj.to(p.device), cutoff=rcut)

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
    ema_net = PEGNN_dynamics(eq_pos=equilibrium_lattice_lj.float(),
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
                        ).to(device).requires_grad_(False)
    ema_net.load_state_dict(torch.load(f"/data1/FEAT-JCP/network_lj_ckpt/egnn_ema_phase{args.phase}_N{args.n_particles}_budget{args.budget}_seed{args.seed}.pth"))


    vector_field_ema = lambda t, x: ema_net(t, x + equilibrium_lattice_lj[None].to(device).float().reshape(-1, n_particles* 3)) 
    score_ema = lambda t, x: (-vector_field_ema(t, x) * (1-t).reshape(-1, 1) / t.reshape(-1, 1) - x / t.reshape(-1, 1)) / (std)**2


       
    # # Controlled Jarzynski 
    def Jarzynski_integrate(
        x1: torch.Tensor, 
        vector_field: callable, 
        score: callable, 
        n_steps: int = 100, 
        forward: bool = True,
        eps: float = 1e-4,
        init_logp = None,
        init_score = None,
        final_logp = None
    ):

        if forward:
            t_start = 0
            t_end = 1 
        else:
            t_start = 1 
            t_end = 0
        
        step_size = (t_end - t_start) / n_steps
        device = x1.device 
        batch_size = x1.shape[0]  

        # initialize
        t = torch.zeros(batch_size).to(device) + t_start
        
        # sample
        x = x1
        std = (np.abs(2 * step_size * eps))**0.5

        w = -init_logp
        vf = vector_field(t, x).detach()
        s = score(t, x).detach()
        vf = remove_mean(vf, n_particles, 3, 0)
        s = remove_mean(s, n_particles, 3, 0)

        for i in tqdm(range(n_steps)):

            if forward:
                if i == 0:
                    # handle numerical instability
                    mean = x.detach() + step_size * vf.detach() + step_size * remove_mean(init_score, n_particles, 3, 0).detach() * eps
                else:
                    mean = x.detach() + step_size * vf.detach() + step_size * s.detach() * eps
                new_x = mean + std * remove_mean(torch.randn_like(x), n_particles, 3, 0)

                w = w - torch.distributions.Normal(mean, std).log_prob(new_x).sum(-1).cpu()
                
                vf_new = vector_field(t+step_size, new_x).detach()
                # s_new = score(t+step_size, new_x).detach()
                if i == n_steps - 1:
                    # calculate final score
                    final_score = []
                    final_logps = []
                    for j in range(new_x.shape[0]):
                        inputs = new_x[j].reshape(-1, n_particles, 3).clone().requires_grad_(True)
                        final_logp_ = final_logp(inputs)
                        final_logps.append(final_logp_)
                        final_score_ = torch.autograd.grad(final_logp_.sum(), inputs, create_graph=False)[0]
                        final_score.append(final_score_)
                    final_score = torch.stack(final_score).reshape(-1, n_particles*3)
                    s_new = final_score
                    
                else:
                    s_new = score(t+step_size, new_x).detach()

                vf_new = remove_mean(vf_new, n_particles, 3, 0)
                s_new = remove_mean(s_new, n_particles, 3, 0)

                mean_new = new_x.detach() - step_size * vf_new.detach() + step_size * s_new.detach() * eps
                w = w + torch.distributions.Normal(mean_new, std).log_prob(x).sum(-1).cpu()
                # print(x)
            else:
                if i == 0:
                    mean = x.detach() + step_size * vf.detach() - step_size * remove_mean(init_score, n_particles, 3, 0).detach() * eps
                else:
                    mean = x.detach() + step_size * vf.detach() - step_size * s.detach() * eps
                new_x = mean + std * remove_mean(torch.randn_like(x), n_particles, 3, 0)

                w = w - torch.distributions.Normal(mean, std).log_prob(new_x).sum(-1).cpu()

                vf_new = vector_field(t+step_size, new_x).detach()
                vf_new = remove_mean(vf_new, n_particles, 3, 0)
                if i == n_steps - 1:
                    # calculate final score
                    final_score = []
                    final_logps = []
                    for j in range(new_x.shape[0]):
                        inputs = new_x[j].reshape(-1, n_particles, 3).clone().requires_grad_(True)
                        final_logp_ = final_logp(inputs)
                        final_logps.append(final_logp_)
                        final_score_ = torch.autograd.grad(final_logp_.sum(), inputs, create_graph=False)[0]
                        final_score.append(final_score_)
                    final_score = torch.stack(final_score).reshape(-1, n_particles*3)
                    s_new = final_score
                    
                else:
                    s_new = score(t+step_size, new_x).detach()
                s_new = remove_mean(s_new, n_particles, 3, 0)
                
                mean_new = new_x.detach() - step_size * vf_new.detach() - step_size * s_new.detach() * eps
                w = w + torch.distributions.Normal(mean_new, std).log_prob(x).sum(-1).cpu()

            t += step_size
            x = new_x
            vf = vf_new
            s = s_new
            # print(x.reshape(-1, n_particles, 3).mean(1))

            x = x.reshape(-1, n_particles, 3) - x.reshape(-1, n_particles, 3).mean(1, keepdim=True)
            x = x.reshape(-1, n_particles * 3)
        print(torch.stack(final_logps).flatten().cpu().shape)
        print(final_score.shape, x.shape)
        w = w + torch.stack(final_logps).flatten().cpu().detach()
        return x, w






    infer_batch_size = min(500, training_size)
    displacements_batch_idx = np.random.choice(training_size, size=training_size, replace=False)
    displacements_batch = displacements[displacements_batch_idx].reshape(training_size, -1).to(device).float()
    gaussian_noise = sample_displacements(training_size).reshape(training_size, -1).to(device).float()

    x = []
    w1 = []
        

    displace_logp = lambda x: -energy_fn_lj(wrap_positions_fn(x.reshape(-1, n_particles, 3) + equilibrium_lattice_lj[None].to(device).float())) / BOLTZMANN_CONSTANT/T


    n_batch = training_size // infer_batch_size
    for i in range(n_batch):
        gaussian_noise_ = gaussian_noise[i*infer_batch_size:(i+1)*infer_batch_size]
        gaussian_noise_ = gaussian_noise_.requires_grad_(True)
        log_p = -energy_func_displacements(gaussian_noise_.reshape(-1, n_particles, 3))/ BOLTZMANN_CONSTANT/T
        init_score = torch.autograd.grad(log_p.sum(), gaussian_noise_, create_graph=False)[0]
        gaussian_noise_ = gaussian_noise_.detach()
        log_p = log_p.detach().cpu()

        x_, w1_ = Jarzynski_integrate(gaussian_noise_, vector_field_ema, score_ema, 500, forward=False, init_logp=log_p, init_score=init_score, final_logp=displace_logp)
        x.append(x_)
        w1.append(w1_)
    x = torch.cat(x, dim=0)
    w1 = torch.cat(w1, dim=0)



    infer_batch_size = min(500, training_size)

    displacements_batch_idx = np.random.choice(training_size, size=training_size, replace=False)
    displacements_batch = displacements[displacements_batch_idx].reshape(training_size, -1).to(device).float()
    gaussian_noise = sample_displacements(training_size).reshape(training_size, -1).to(device).float()
        


    displace_logp = lambda x: -energy_fn_lj(wrap_positions_fn(x.reshape(-1, n_particles, 3) + equilibrium_lattice_lj[None].to(device).float()))/ BOLTZMANN_CONSTANT/T

    x = []
    w2 = []

    n_batch = training_size // infer_batch_size
    for i in range(n_batch):
        displacements_batch_ = displacements_batch[i*infer_batch_size:(i+1)*infer_batch_size]

        init_score = []
        log_p = []
        for i in tqdm(range(displacements_batch_.shape[0])):
            inputs = displacements_batch_.reshape(-1, n_particles* 3)[i].requires_grad_(True)
            # calculate score 
            log_p_ = displace_logp(inputs)
            log_p.append(log_p_)

            init_score_ = torch.autograd.grad(log_p_.sum(), inputs, create_graph=False)[0]
            init_score.append(init_score_)


        init_score = torch.stack(init_score).reshape(-1, n_particles*3)
        log_p = torch.stack(log_p).flatten().detach().cpu()

        x_, w2_ = Jarzynski_integrate(displacements_batch_, vector_field_ema, score_ema, 500, forward=True, init_logp=log_p, init_score=init_score, final_logp=lambda x:  -energy_func_displacements(x.reshape(-1, n_particles, 3))/ BOLTZMANN_CONSTANT/T)
        x.append(x_)
        w2.append(w2_)
    x = torch.cat(x, dim=0)
    w2 = torch.cat(w2, dim=0)


    DF1 = torch.logsumexp(w1, 0).item() - np.log(w1.shape[0])
    DF2 = -torch.logsumexp(w2, 0).item() + np.log(w2.shape[0])     
    DF_lower = w1.mean().item()
    DF_upper = -w2.mean().item()
    mv1 = (DF1 + DF2) / 2

    for iii in range(10000):
        mv1_new = (torch.logsumexp(torch.nn.LogSigmoid()(w1 - mv1), 0) - torch.logsumexp(torch.nn.LogSigmoid()(w2 + mv1), 0) + mv1).item()
        if np.abs(mv1_new - mv1) < 1e-4:
            mv1 = mv1_new
            break
        mv1 = mv1_new
    
    if args.n_particles == 108:
        absolute_free_energy = 6.109948647123855
    if args.n_particles == 180:
        absolute_free_energy = 6.153864796495445
    if args.n_particles == 256:
        absolute_free_energy = 6.174516817493909
    if args.n_particles == 500:
        absolute_free_energy = 6.199770354466455
    if args.test_budget == 'low':
        save_dir = 'results100'
    if args.test_budget == 'medium':
        save_dir = 'results1000'
    if args.test_budget == 'high':
        save_dir = 'results10k'
    with open(f"/data1/FEAT-JCP/{save_dir}/lj_egnn_system{args.phase}_{args.n_particles}_budget{args.budget}_seed{args.seed}.txt", "w") as f:
        f.write(f'Two sides: {mv1/n_particles - absolute_free_energy}\n')
        f.write(f'Prior side: {DF1/n_particles - absolute_free_energy}\n')
        # ESS
        _w = torch.nn.Softmax()(w1)
        ESS =  _w.sum()**2/(_w**2).sum()
        f.write(f'FWD ESS: {ESS}\n')
        _w = torch.nn.Softmax()(w2)
        ESS =  _w.sum()**2/(_w**2).sum()
        f.write(f'BWD ESS: {ESS}\n')

    plt.hist(w1.cpu().numpy().flatten()/n_particles, bins=100, color='skyblue', edgecolor='k', alpha=0.5)
    plt.hist(-w2.cpu().numpy().flatten()/n_particles, bins=100, color='red', edgecolor='k', alpha=0.5)
    plt.savefig(f"/data1/FEAT-JCP/figures/lj_egnn_system{args.phase}_{args.n_particles}_budget{args.budget}_seed{args.seed}_histogram.png")
    plt.close()