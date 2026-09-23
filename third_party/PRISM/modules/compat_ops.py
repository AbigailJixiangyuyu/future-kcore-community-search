"""Small-graph PyTorch operators; no global monkey patches or CUDA extensions."""

import math
import torch
from torch import nn

try:
    from torch_scatter import scatter as _native_scatter
except ModuleNotFoundError:
    _native_scatter = None


def zeros(value):
    if value is not None:
        with torch.no_grad():
            value.zero_()


def isin(elements, candidates):
    return (elements.unsqueeze(-1) == candidates.reshape(-1)).any(-1)


def scatter_sum(src, index, dim=0, out=None, dim_size=None):
    if dim != 0 or index.dim() != 1:
        raise NotImplementedError("Compatibility scatter supports a 1-D index on dim=0")
    size = int(index.max()) + 1 if index.numel() else 0
    size = size if dim_size is None else dim_size
    if out is None:
        out = src.new_zeros((size,) + src.shape[1:])
    return out.index_add_(0, index, src)


scatter_add = scatter_sum


def scatter_max(src, index, dim=0, out=None, dim_size=None):
    if dim != 0 or src.dim() != 1 or out is not None:
        raise NotImplementedError("Compatibility scatter_max supports 1-D input")
    size = int(index.max()) + 1 if index.numel() else 0
    size = size if dim_size is None else dim_size
    values, positions = [], []
    for group in range(size):
        rows = (index == group).nonzero(as_tuple=False).reshape(-1)
        if rows.numel():
            value, pos = src[rows].max(0)
            values.append(value)
            positions.append(rows[pos])
        else:
            values.append(src.new_zeros(()))
            positions.append(index.new_tensor(src.numel()))
    if not size:
        return src.new_empty(0), index.new_empty(0)
    return torch.stack(values), torch.stack(positions)


def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
    if _native_scatter is not None:
        return _native_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)
    if reduce == "max":
        return scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
    result = scatter_sum(src, index, dim=dim, dim_size=dim_size)
    if reduce == "mean":
        counts = scatter_sum(src.new_ones(index.numel()), index,
                             dim_size=result.size(0)).clamp(min=1)
        return result / counts.reshape((-1,) + (1,) * (src.dim() - 1))
    if reduce not in ("sum", "add"):
        raise NotImplementedError(reduce)
    return result


def scatter_softmax(src, index, dim=0):
    if dim != 0:
        raise NotImplementedError("Only dim=0 is supported")
    result = torch.zeros_like(src)
    for group in index.unique():
        mask = index == group
        result[mask] = torch.softmax(src[mask], dim=0)
    return result


def recent_index(ei_src, ei_dst, pos_node_s, pos_node_d, batch_size):
    """Match recent_index_kernel in mem_update_graph.cu, including src-first ties."""
    result = torch.full_like(ei_src, -1)
    for j in range(batch_size):
        eligible = j < ei_dst.remainder(batch_size)
        result = torch.where(eligible & (ei_src == pos_node_d[j]),
                             torch.full_like(result, j + batch_size), result)
        result = torch.where(eligible & (ei_src == pos_node_s[j]),
                             torch.full_like(result, j), result)
    return result


class TransformerConv(nn.Module):
    """Default PRISM TransformerConv contract, not a general PyG replacement.

    Multi-head dot-product attention with edge keys/values, concatenated heads
    and a learned root skip. Parameter naming follows the default PyG operator.
    """

    def __init__(self, in_channels, out_channels, heads=2, dropout=0.1,
                 edge_dim=None):
        super().__init__()
        self.heads, self.out_channels, self.dropout = heads, out_channels, dropout
        width = heads * out_channels
        self.lin_key = nn.Linear(in_channels, width)
        self.lin_query = nn.Linear(in_channels, width)
        self.lin_value = nn.Linear(in_channels, width)
        self.lin_edge = nn.Linear(edge_dim, width, bias=False)
        self.lin_skip = nn.Linear(in_channels, width)

    def forward(self, x, edge_index, edge_attr):
        source, target = edge_index
        shape = (-1, self.heads, self.out_channels)
        query = self.lin_query(x).view(shape)[target]
        edge = self.lin_edge(edge_attr).view(shape)
        key = self.lin_key(x).view(shape)[source] + edge
        value = self.lin_value(x).view(shape)[source] + edge
        logits = (query * key).sum(-1) / math.sqrt(self.out_channels)
        alpha = scatter_softmax(logits, target)
        alpha = nn.functional.dropout(alpha, self.dropout, self.training)
        messages = (alpha.unsqueeze(-1) * value).reshape(
            -1, self.heads * self.out_channels)
        return scatter_sum(messages, target, dim_size=x.size(0)) + self.lin_skip(x)
