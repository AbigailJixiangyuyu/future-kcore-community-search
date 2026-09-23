import torch
from torch import Tensor

try:
    from torch_geometric.utils import scatter
except ImportError:
    from .compat_ops import scatter


class MeanAggregator(torch.nn.Module):
    def __init__(self, emb_dim: int = 100):
        super().__init__()

    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int, inverse_indices: Tensor = None):
        if inverse_indices is None:
            return scatter(msg, index, dim=0, dim_size=dim_size, reduce="mean")
        else:
            return scatter(msg[inverse_indices], index, dim=0,
                           dim_size=dim_size, reduce="mean")
