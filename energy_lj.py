import torch
from torch.func import vmap, grad
from typing import Optional
from typing import Optional, Callable
from energy_mw import pairwise_squared_distance_pbc
Array = torch.Tensor

# -------------------------------
# Lennard-Jones parameters
# -------------------------------
LJ_EPSILON =  0.238 # depth of potential well in kcal / mole
LJ_SIGMA = 3.4    # distance at which potential crosses zero
# LJ_CUTOFF = 2.7 * LJ_SIGMA   # cutoff radius in sigma units
LJ_SHIFT_ENERGY = True  # shift potential to zero at cutoff

# -------------------------------
# Helpers from the monoatomic water file:
#   - pairwise_difference_pbc
#   - pairwise_squared_distance_pbc
# -------------------------------
# We assume they are already defined or imported:
# from your_mw_file import pairwise_difference_pbc, pairwise_squared_distance_pbc

# Precompute shift value if needed
def _lj_energy_unshifted(r2: torch.Tensor) -> torch.Tensor:
    inv_r2 = (LJ_SIGMA ** 2) / r2
    inv_r6 = inv_r2 ** 3
    return 4.0 * LJ_EPSILON * (inv_r6 ** 2 - inv_r6)


def energy_lj_single_sample(coordinates: Array, box_length: Array, reduced_cutoff: float) -> Array:
    """
    Batched-safe Lennard-Jones energy for a single configuration.
    coordinates: (N, D)
    box_length: (D,)
    """
    N = coordinates.shape[0]

    cutoff = reduced_cutoff * LJ_SIGMA

    # Pairwise squared distances with PBC
    r2 = pairwise_squared_distance_pbc(coordinates, box_length)

    # Avoid self-interactions by adding +inf on diagonal
    r2 = r2 + torch.eye(N, device=r2.device) * 1e10

    # Compute LJ energy
    inv_r2 = (LJ_SIGMA ** 2) / r2
    inv_r6 = inv_r2 ** 3
    e = 4.0 * LJ_EPSILON * (inv_r6 ** 2 - inv_r6)
    
    shift_val = _lj_energy_unshifted(torch.tensor(cutoff ** 2)) if LJ_SHIFT_ENERGY else torch.tensor(0.0)
    mask = r2 <= (cutoff ** 2)
    e = torch.where(mask, e - shift_val.to(e.device), torch.tensor(0.0, device=e.device))

    return torch.sum(torch.triu(e, diagonal=1))

def energy_lj(coordinates: Array, box_length: Array, cutoff: float) -> Array:
    """
    Batched Lennard-Jones energy.
    coordinates: (B, N, D) or (N, D)
    box_length: (B, D) or (D,)
    Returns: (B,) or scalar if single config
    """
    # Ensure batch dimension
    if coordinates.ndim == 2:
        coordinates = coordinates.unsqueeze(0)  # (1, N, D)
        single = True
    else:
        single = False

    B, N, D = coordinates.shape

    # Broadcast box_length
    if box_length.ndim == 1:
        box = box_length.expand(B, -1)  # (B, D)
    else:
        box = box_length  # (B, D)

    # vmap over batch
    energies = vmap(lambda c, b: energy_lj_single_sample(c, b, cutoff))(coordinates, box)

    if single:
        energies = energies.squeeze(0)

    return energies

def forces_lj(coordinates: Array, box_length: Array, cutoff: float) -> Array:
    """
    Batched forces computation, supports multiple configurations at once.
    coordinates: (B, N, D) or (N, D)
    box_length: (B, D) or (D,)
    Returns: (B, N, D) or (N, D)
    """
    # Ensure batch dimension
    if coordinates.ndim == 2:
        coordinates = coordinates.unsqueeze(0)  # (1, N, D)
        single = True
    else:
        single = False

    B, N, D = coordinates.shape

    # Broadcast box_length
    if box_length.ndim == 1:
        box = box_length.expand(B, -1)  # (B, D)
    else:
        box = box_length  # (B, D)

    coordinates = coordinates.detach().requires_grad_()

    def grad_fn(c, b):
        return grad(lambda x: energy_lj_single_sample(x, b, cutoff))(c)

    forces = -vmap(grad_fn)(coordinates, box)  # shape (B, N, D)

    if single:
        forces = forces.squeeze(0)

    return forces
