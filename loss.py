import torch
from utils import Coef, remove_mean


def v_loss(x1: torch.Tensor, 
           x2: torch.Tensor, 
           alpha: Coef,
           beta: Coef,
           vector_field: callable):
    device = x1.device
    batch_size = x1.shape[0]
    dim_size = [1 for _ in x1.shape[1:]]

    t = torch.clip(torch.rand(batch_size).to(device), 1e-4, 1-1e-4)

    xi = alpha.t(t).reshape(-1, *dim_size) * x1 + beta.t(t).reshape(-1, *dim_size) * x2 
    dt_xi = alpha.dt(t).reshape(-1, *dim_size) * x1 + beta.dt(t).reshape(-1, *dim_size) * x2
    b_t = vector_field(t, xi)
    target = dt_xi
#     print(target.shape, b_t.shape)
    loss = torch.nn.MSELoss()(target, b_t)

    return loss.mean()
