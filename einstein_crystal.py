import math
from dataclasses import dataclass, field
import numpy as np
import torch
from torch.utils.data import IterableDataset
from src.manifolds.flat_torus import FlatTorus as FT_cubic
from src.manifolds.flat_torus_triclinic import FlatTorus as FT_triclinic
from src.data.distribution import DistributionDataset, DistributionGenerated
from src.utils import ensure_tensor
from typing import Optional, Union, List

@dataclass
class EinsteinCrystal:
    """
    Classical Einstein crystal with identical spring constants.

    The model treats every particle as tethered to a fixed lattice site
    by a harmonic spring of force constant *k* at inverse temperature
    *beta*.  Any two of the parameters (`k`, `beta`, `sigma`) determine
    the third through::

        σ = (2 β k)^{-1/2}

    Parameters
    ----------
    k : float, optional
        Spring constant (energy / length²).  Set *None* to infer it.
    beta : float, optional
        Inverse temperature 1 / (k_B T).  Set *None* to infer it.
    sigma : float, optional
        Cartesian displacement s.d.; set *None* to infer it.
    eq_pos : tensor | str | list[str]
        (N x 3) tensor of equilibrium positions (or a path / list(path, key)).
    generator : torch.Generator, optional
        RNG for reproducible sampling.

    Notes
    -----
    * Geometry is handled with periodic boundary conditions via supplied
    manifold.
    """

    k: Optional[float] = None
    beta: Optional[float] = None
    sigma: Optional[float] = None
    lambda_de_broglie: float = 1.0
    eq_pos: Union[torch.Tensor, str, List[str]] = field(
        default_factory=lambda: torch.eye(3) * 0.5
    )
    manifold: Union[FT_cubic, FT_triclinic] = FT_cubic(n_particles=1, s_dim=1)
    generator: Optional[torch.Generator] = None
    fix_com: bool = False
    first_particle_removed: bool = False

    def __post_init__(self):
        # Deduce the missing parameter -------------------------
        if self.k is not None and self.beta is not None and self.sigma is None:
            self.sigma = (1.0 / (2 * self.beta * self.k)) ** 0.5
        elif self.k is None and self.beta is not None and self.sigma is not None:
            self.k = 1.0 / (2 * self.beta * self.sigma ** 2)
        elif self.k is not None and self.beta is None and self.sigma is not None:
            self.beta = 1.0 / (2 * self.k * self.sigma ** 2)
        else:
            assert abs((1.0 / (2 * self.beta * self.k)) ** 0.5 - self.sigma) < 1e-6

        # Load / coerce equilibrium positions --------------------------------
        if isinstance(self.eq_pos, str):
            self.eq_pos = np.load(self.eq_pos)["config"]
        elif isinstance(self.eq_pos, list):
            self.eq_pos = np.load(self.eq_pos[0])[self.eq_pos[1]]
        self.eq_pos = ensure_tensor(self.eq_pos)

        n_particles, s_dim = self.eq_pos.shape
        expected = (self.manifold.n_particles, self.manifold.s_dim)
        if (n_particles, s_dim) != expected:
            raise ValueError(
            f"eq_pos shape mismatch: got {(n_particles, s_dim)}, "
            f"expected {expected} as for manifold {type(self.manifold).__name__}"
            )

    def sample_displacements(self, n_samples):
        disp = self.sigma * torch.randn(n_samples, *self.eq_pos.shape, generator=self.generator)
        if self.fix_com is False:
            return disp
        return disp - disp.mean(dim=1).unsqueeze(1)

    def sample_positions(self, n_samples):
        positions = self.sample_displacements(n_samples=n_samples) + self.eq_pos
        positions = self.manifold.projx(positions.flatten(start_dim=1))
        return positions.reshape(-1, self.manifold.n_particles, self.manifold.s_dim)

    def energy_func_displacements(self, x):
        return torch.sum(self.k * x**2, dim=(-2, -1))

    def energy_func_positions(self, x):
        displacements = self.manifold.logmap(x.flatten(start_dim=1), self.eq_pos.flatten().to(x))
        displacements = displacements.reshape(-1, self.manifold.n_particles, self.manifold.s_dim)
        return self.energy_func_displacements(displacements)
    
    @property
    def abs_free_energy(self):
        """Returns beta / N * F for the Einstein crystal following Wirnsberger et al. (SI)"""
        n_particles = torch.tensor(self.manifold.n_particles)
        einstein_spring_constant = torch.tensor(self.k)
        beta = torch.tensor(self.beta)
        if hasattr(self.manifold, 'box_vectors'):
            volume = torch.abs(torch.det(self.manifold.box_vectors)).item()
        else:
            volume = self.manifold.box**3

        first_term = 1. / n_particles * torch.log(n_particles * self.lambda_de_broglie**3. / volume)

        if self.fix_com is True:
            second_term = 3./2. * (1 - 1. / n_particles) * torch.log(beta * einstein_spring_constant * self.lambda_de_broglie**2. / torch.pi)
            third_term = -3. / (2. * n_particles) * torch.log(n_particles)
        else:
            second_term = 3./2. * torch.log(beta * einstein_spring_constant * self.lambda_de_broglie**2. / torch.pi)
            third_term = 0

        return first_term + second_term + third_term
        
    
    def energy_func(self, x):
        return self.energy(x)


class EinsteinCrystalDisplacementsInfinite(EinsteinCrystal, IterableDataset, DistributionGenerated):
    """Important: Set Dataloader(batch_size=None), as iter already yields batches"""
    def __init__(
        self,
        k=None,
        beta=None,
        sigma=None,
        eq_pos=torch.ones(1, 3) * 0.5,
        manifold=FT_cubic(n_particles=1, s_dim=1),
        lambda_de_broglie=1.0,
        generator=None,
        fix_com=False,
        first_particle_removed=False,
        sample_size=1,
        **kwargs,
    ):
        super().__init__(
            k=k,
            beta=beta,
            sigma=sigma,
            eq_pos=eq_pos,
            manifold=manifold,
            lambda_de_broglie=lambda_de_broglie,
            generator=generator,
            fix_com=fix_com,
            first_particle_removed=first_particle_removed,
            **kwargs,
        )
        self.sample_size = sample_size

    def __iter__(self):
        while True:
            yield self.sample_displacements(self.sample_size)

    def sample(self, n_samples):
        return self.sample_displacements(n_samples)

    def energy(self, samples):
        return self.energy_func_displacements(x=samples)


class EinsteinCrystalPositionsInfinite(EinsteinCrystal, IterableDataset, DistributionGenerated):
    """Important: Set Dataloader(batch_size=None), as iter already yields batches"""
    def __init__(
        self,
        k=None,
        beta=None,
        sigma=None,
        eq_pos=torch.ones(1, 3) * 0.5,
        manifold=FT_cubic(n_particles=1, s_dim=1),
        lambda_de_broglie=1.0,
        generator=None,
        fix_com=False,
        first_particle_removed=False,
        sample_size=1,
        **kwargs,
    ):
        super().__init__(
            k=k,
            beta=beta,
            sigma=sigma,
            eq_pos=eq_pos,
            manifold=manifold,
            lambda_de_broglie=lambda_de_broglie,
            generator=generator,
            fix_com=fix_com,
            first_particle_removed=first_particle_removed,
            **kwargs,
        )
        self.sample_size = sample_size

    def __iter__(self):
        while True:
            yield self.sample_positions(self.sample_size)

    def sample(self, n_samples):
        return self.sample_positions(n_samples)

    def energy(self, samples):
        return self.energy_func_positions(x=samples)


class EinsteinCrystalDisplacementsFinite(EinsteinCrystal, DistributionDataset):
    
    def __init__(
        self,
        k=None,
        beta=None,
        sigma=None,
        eq_pos=torch.ones(1, 3) * 0.5,
        manifold=FT_cubic(n_particles=1, s_dim=1),
        lambda_de_broglie=1.0,
        generator=None,
        fix_com=False,
        first_particle_removed=False,
        ds_size=1000,
        **kwargs,
    ):
        super().__init__(
            k=k,
            beta=beta,
            sigma=sigma,
            eq_pos=eq_pos,
            manifold=manifold,
            lambda_de_broglie=lambda_de_broglie,
            generator=generator,
            fix_com=fix_com,
            first_particle_removed=first_particle_removed,
            **kwargs,
        )
        self.displacements = self.sample_displacements(ds_size)
        self.energies = self.energy_func_displacements(self.displacements)

        torch.allclose(self.energies[:2], self.energy(self.data[:2]))

    @property
    def data(self):
        return self.displacements

    def energy(self, samples):
        return self.energy_func_displacements(x=samples)


class EinsteinCrystalPositionsFinite(EinsteinCrystal, DistributionDataset):

    def __init__(
        self,
        k=None,
        beta=None,
        sigma=None,
        eq_pos=torch.ones(1, 3) * 0.5,
        manifold=FT_cubic(n_particles=1, s_dim=1),
        lambda_de_broglie=1.0,
        generator=None,
        fix_com=False,
        first_particle_removed=False,
        ds_size=1000,
        **kwargs,
    ):
        super().__init__(
            k=k,
            beta=beta,
            sigma=sigma,
            eq_pos=eq_pos,
            manifold=manifold,
            lambda_de_broglie=lambda_de_broglie,
            generator=generator,
            fix_com=fix_com,
            first_particle_removed=first_particle_removed,
            **kwargs,
        )
        self.positions = self.sample_positions(n_samples=ds_size)
        self.energies = self.energy_func_positions(self.positions)

        # torch.allclose(self.energies[0], self.energy(self.data[0]))
        torch.allclose(self.energies[:2], self.energy(self.data[:2]))

    @property
    def data(self):
        return self.positions

    def energy(self, samples):
        return self.energy_func_positions(x=samples)
