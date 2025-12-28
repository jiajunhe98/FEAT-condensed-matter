import torch
import math
from typing import Optional
from torch.func import vmap
from torch.func import grad

Array = torch.Tensor

MW_A = 7.049556277
MW_B = 0.6022245584
MW_GAMMA = 1.2
MW_EPSILON = 6.189  # Kcal/mol
MW_SIGMA = 2.3925  # Angstrom  ## what is this?
MW_REDUCED_CUTOFF = 1.8
MW_COS = math.cos(109.47 / 180.0 * math.pi)
MW_LAMBDA = 23.15

def wrap_differences(coordinate_deltas: torch.Tensor, box_length: torch.Tensor) -> torch.Tensor:
    """
    Wrap coordinate differences into the primary image using periodic boundary conditions.

    Args:
        coordinate_deltas: Tensor of shape [..., dim]
        box_length: Tensor of shape [..., dim], must be broadcastable with coordinate_deltas

    Returns:
        Tensor of shape [..., dim] with differences wrapped into the primary box.
    """
    return coordinate_deltas - box_length * torch.round(coordinate_deltas / box_length)





def wrap_coordinates(coordinate: torch.Tensor, box_length: torch.Tensor, lower: torch.Tensor) -> torch.Tensor:
    """
    Wrap coordinate differences into the primary image using periodic boundary conditions.

    Args:
        coordinate_deltas: Tensor of shape [..., dim]
        box_length: Tensor of shape [..., dim], must be broadcastable with coordinate_deltas

    Returns:
        Tensor of shape [..., dim] with differences wrapped into the primary box.
    """
    return torch.remainder(coordinate - lower, box_length) + lower


def energy_mw_single_sample(coordinates, box_length=1):
    """ Compute the energy of a configuration of monoatomic water with pbc"""

    dr = pairwise_difference_pbc(coordinates, box_length)
    dr /= MW_SIGMA

    def _three_body_energy(dr: Array) -> Array:
        """Compute three-body term for one sample.

        Args:
        dr: [num_particles, num_particles, 3] array of distance vectors
            between the particles.
        Returns:
        The three-body energy contribution of the sample (a scalar).
        """

        def _one_particle_contribution(dri: Array) -> Array:
            # dri is (num_particles-1) x 3.
            raw_norms = torch.linalg.norm(dri, axis=-1)
            keep = raw_norms < MW_REDUCED_CUTOFF
            norms = torch.where(keep, raw_norms, 1e20)
            norm_energy = torch.exp(MW_GAMMA / (norms - MW_REDUCED_CUTOFF))
            norm_energy = torch.where(keep, norm_energy, 0.0)
            normprods = norms[None, :] * norms[:, None]
            # Note: the sum below is equivalent to:
            # dotprods = torch.dot(dri, dri[..., None]).squeeze(-1)
            # but using torch.dot results in loss of precision on TPU,
            # as evaluated by comparing to MD samples.
            dotprods = torch.sum(dri[:, None, :] * dri[None, :, :], axis=-1)

            cos_ijk = dotprods / normprods

            energy = MW_LAMBDA * MW_EPSILON * (MW_COS - cos_ijk) ** 2
            energy *= norm_energy
            energy = torch.triu(energy, 1)
            energy = torch.sum(energy, axis=-1)
            return torch.dot(energy, norm_energy)

        # Remove diagonal elements [i, i, :], changing the shape from
        num_particles = dr.shape[0]
        mask = torch.eye(num_particles, dtype=torch.bool, device=dr.device).unsqueeze(-1)  # (N, N, 1)

        clean_dr = dr +mask * 1e6
        # Vectorize over particles.
        #energy = torch.sum(torch.func.vmap(_one_particle_contribution)(clean_dr))
        #print(clean_dr.shape)
        energy = torch.sum(torch.vmap(_one_particle_contribution)(clean_dr))

        # energy = 0
        # for dri in clean_dr:
        #     energy += _one_particle_contribution(dri)
        return energy

    def _two_body_energy(r2: Array) -> Array:
        r2 /= MW_SIGMA**2
        r = torch.sqrt(r2)
        mask = r < MW_REDUCED_CUTOFF
        # Distances on or above the cutoff can cause NaNs in the gradient of
        # `term_2` below, even though they're masked out in the forward computation.
        # To avoid this, we set these distances to a safe value.
        r = torch.where(mask, r, 2.0 * MW_REDUCED_CUTOFF)
        term_1 = MW_A * MW_EPSILON * (MW_B / r2**2 - 1.0)
        term_2 = torch.where(mask, torch.exp(1.0 / (r - MW_REDUCED_CUTOFF)), 0.0)
        energy = term_1 * term_2
        return energy

    # Vectorize over samples.
    three_body_energy = _three_body_energy(dr)

    r2 = pairwise_squared_distance_pbc(coordinates, box_length)
    r2 += torch.eye(r2.shape[-1]).to(coordinates.device)

    energies = _two_body_energy(r2)
    two_body_energy = torch.sum(torch.triu(energies, diagonal=1), axis=[-2, -1])

    #print("Two-body contribution:", two_body_energy)
    #print("Three-body contribution:", three_body_energy)

    energy = three_body_energy + two_body_energy
    return energy


def pairwise_difference_pbc(coordinates: Array, box_length: Array) -> Array:
    """Computes pairwise distance vectors obeying periodic boundary conditions.

    Args:
      coordinates: array with shape [..., num_particles, dim] containing particle
        coordinates.
      box_length: array with shape [..., dim], the edge lengths of the box.

    Returns:
      Array with shape [..., num_particles, num_particles, dim], the pairwise
      distance vectors with respect to periodic boundary conditions.
    """
    deltas = _pairwise_difference(coordinates)
    # chex.assert_is_broadcastable(box_length.shape[:-1], coordinates.shape[:-2])
    box_length = box_length[..., None, None, :]
    return deltas - box_length * torch.round(deltas / box_length)


def _pairwise_difference(coordinates: Array) -> Array:
    """Computes pairwise difference vectors.

    Args:
      coordinates: array with shape [..., num_particles, dim] containing particle
        coordinates.

    Returns:
      Array with shape [..., num_particles, num_particles, dim], difference
      vectors for all pairs of particles.
    """
    if coordinates.ndim < 2:
        raise ValueError(
            f"Expected at least 2 array dimensions, got {coordinates.ndim}."
        )
    x = coordinates[..., :, None, :]
    y = coordinates[..., None, :, :]
    return x - y


def pairwise_squared_distance_pbc(coordinates: Array, box_length: Array) -> Array:
    """Computes pairwise squared distance obeying periodic boundary conditions.

    Args:
      coordinates: array with shape [..., num_particles, dim] containing particle
        coordinates.
      box_length: array with shape [..., dim], the edge lengths of the box.

    Returns:
      Array with shape [..., num_particles, num_particles] with pairwise squared
      distances.
    """
    coordinate_deltas = _pairwise_difference(coordinates)
    # chex.assert_is_broadcastable(box_length.shape[:-1], coordinates.shape[:-2])
    return squared_distance_pbc(coordinate_deltas, box_length[..., None, None, :])


def squared_distance_pbc(coordinate_deltas: Array, box_length: Array) -> Array:
    """Computes squared distance obeying periodic boundary conditions.

    Args:
      coordinate_deltas: array with shape [..., dim] containing difference
        vectors.
      box_length: array with shape [..., dim], the edge lengths of the box.

    Returns:
      Array with shape [...], the squared distances with respect to periodic
      boundary conditions.
    """
    # chex.assert_is_broadcastable(box_length.shape[:-1], coordinate_deltas.shape[:-1])
    coordinate_deltas_pbc = coordinate_deltas - box_length * torch.round(
        coordinate_deltas / box_length
    )
    return torch.sum(coordinate_deltas_pbc**2, axis=-1)

import torch
from torch.func import vmap

def energy_mw(coordinates: torch.Tensor, box_length: torch.Tensor):
    """
    Wrapper for energy_mw_single_sample that supports both single and batched coordinates.
    Also accepts a single 1D box_length to be broadcast across batch.

    Args:
        coordinates: Tensor of shape (..., N, D)
        box_length: Tensor of shape (..., D) or (D,) to be broadcast

    Returns:
        energies: Tensor of shape (...)
    """
    batch_shape = coordinates.shape[:-2]
    N, D = coordinates.shape[-2:]

    coords_flat = coordinates.reshape(-1, N, D)

    # Handle shared box
    if box_length.ndim == 1:
        box_flat = box_length.expand(coords_flat.shape[0], -1)
    else:
        box_flat = box_length.reshape(-1, D)

    energies_flat = vmap(energy_mw_single_sample)(coords_flat, box_flat)
    energies = energies_flat.reshape(batch_shape)

    return energies

def forces_mw(
    coordinates: torch.Tensor,
    box_length: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes forces as negative gradient of mw_energy w.r.t coordinates.
    Accepts both batched and single box input.

    Args:
        coordinates: Tensor of shape (..., N, D)
        box_length: Tensor of shape (..., D) or (D,) or None

    Returns:
        forces: Tensor of shape (..., N, D)
    """
    batch_dims = coordinates.shape[:-2]
    N, D = coordinates.shape[-2:]
    coords_flat = coordinates.reshape(-1, N, D)

    # Default box: zero (i.e., no PBC)
    if box_length is None:
        box_flat = torch.zeros((coords_flat.shape[0], D), device=coordinates.device)
    elif box_length.ndim == 1:
        box_flat = box_length.expand(coords_flat.shape[0], -1)
    else:
        box_flat = box_length.reshape(-1, D)

    coords_flat = coords_flat.detach().requires_grad_()

    energy_grad = grad(energy_mw_single_sample)
    forces_flat = -vmap(energy_grad)(coords_flat, box_flat)

    return forces_flat.reshape(*batch_dims, N, D)