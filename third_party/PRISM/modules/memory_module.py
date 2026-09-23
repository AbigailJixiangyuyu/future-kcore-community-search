import copy
from typing import Callable, Dict, Tuple

import torch
from torch import Tensor
from torch.nn import GRUCell, RNNCell

try:
    from torch_geometric.nn.inits import zeros
    from torch_geometric.utils import scatter
except ImportError:
    from .compat_ops import zeros, scatter
from .compat_ops import isin, recent_index

from .time_enc import TimeEncoder

# import pdb
# import time
try:
    import mem_update_graph
except ModuleNotFoundError:
    mem_update_graph = None
try:
    from torch_scatter import scatter_max
except ModuleNotFoundError:
    from .compat_ops import scatter_max

TGNMessageStoreType = Dict[int, Tuple[Tensor, Tensor, Tensor, Tensor]]




class DAATGNMemory(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        message_module: Callable,
        aggregator_module: Callable,
        memory_updater_cell: str = "gru",
        layer: int = 3,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.layer = layer

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        # self.gru = GRUCell(message_module.out_channels, memory_dim)
        if memory_updater_cell == "gru":  # for TGN
            self.memory_updater = GRUCell(message_module.out_channels, memory_dim)
        elif memory_updater_cell == "rnn":  # for JODIE & DyRep
            self.memory_updater = RNNCell(message_module.out_channels, memory_dim)
        else:
            raise ValueError(
                "Undefined memory updater!!! Memory updater can be either 'gru' or 'rnn'."
            )

        self.register_buffer("memory", torch.empty(num_nodes, memory_dim))
        last_update = torch.empty(self.num_nodes, dtype=torch.long)
        self.register_buffer("last_update", last_update)
        self.register_buffer("_assoc", torch.empty(num_nodes, dtype=torch.long))

        self.msg_s_store = {}
        self.msg_d_store = {}

        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return self.time_enc.lin.weight.device

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if hasattr(self.msg_s_module, "reset_parameters"):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, "reset_parameters"):
            self.msg_d_module.reset_parameters()
        if hasattr(self.aggr_module, "reset_parameters"):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.memory_updater.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Resets the memory to its initial state."""
        zeros(self.memory)
        zeros(self.last_update)
        self._reset_message_store()

    def detach(self):
        """Detaches the memory from gradient computation."""
        self.memory.detach_()
        
    def mem_graph(self, ei_src, ei_dst, pos_node_s, pos_node_d):
        batch_size = pos_node_s.size(0)
        if ei_src.numel() == 0:
            return torch.empty_like(ei_src)
        if mem_update_graph is None or not ei_src.is_cuda:
            return recent_index(ei_src, ei_dst, pos_node_s, pos_node_d, batch_size)
        # Validation batches live on CPU; the native kernel reads GPU pointers.
        return mem_update_graph.mem_graph(
            ei_src.to(torch.int64).contiguous(),
            ei_dst.to(device=ei_src.device, dtype=torch.int64).contiguous(),
            pos_node_s.to(device=ei_src.device, dtype=torch.int64).contiguous(),
            pos_node_d.to(device=ei_src.device, dtype=torch.int64).contiguous(),
            batch_size
        )

    # def prep(self, ei_src, ei_dst, pos_node_s, pos_node_d):
    #     batch_size = pos_node_s.size(0)
    #     recent_indices = []
    #     # breakpoint()
    #     for i in range(ei_src.size(0)):
    #         target_node = ei_src[i].item()
    #         max_idx = ei_dst[i].item() % batch_size

    #         found_idx = -1
    #         for j in reversed(range(min(max_idx, pos_node_s.size(0)))):
    #             if pos_node_s[j].item() == target_node:
    #                 found_idx = j
    #                 break
    #             if pos_node_d[j].item() == target_node:
    #                 found_idx = j+batch_size
    #                 break
    #         if found_idx == -1:
    #             breakpoint()
    #         recent_indices.append(found_idx)

    #     return torch.tensor(recent_indices, device=ei_dst.device)

        # return None

    def forward(self, n_id, b_edge_index, b_t, b_raw_msg, b_isrc, delivery_addr = None) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        memory, last_update = self._get_updated_memory(n_id)
            # return self._apply_intra_batch_info(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)
        # breakpoint()
        memory = memory[self._assoc[n_id]]
        init_mem = None # memory
        for _ in range(self.layer-1):
            memory, last_update_n =  self._apply_intra_batch_info_v2(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc, init_mem = init_mem, delivery_addr = delivery_addr)
        return self._apply_intra_batch_info_v2(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc, init_mem = init_mem, delivery_addr = delivery_addr)

        # if self.training:
        #     memory, last_update = self._get_updated_memory(n_id)
        #     # return self._apply_intra_batch_info(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)
        
        #     memory = memory[self._assoc[n_id]]
        #     for _ in range(self.layer-1):
        #         memory, last_update_n =  self._apply_intra_batch_info_v2(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)
        #     return self._apply_intra_batch_info_v2(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)
        # else:
        #     nn_id  = n_id.unique()
        #     self._assoc[nn_id] = torch.arange(nn_id.size(0), device=nn_id.device)

        #     memory, last_update = self.memory[nn_id], self.last_update[nn_id]

        #     memory = memory[self._assoc[n_id]]
        #     for _ in range(self.layer-1):
        #         memory, last_update_n =  self._apply_intra_batch_info_v2(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)
        #     return self._apply_intra_batch_info_v2(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)

            # return self._apply_intra_batch_info(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)

        # return memory, last_update

    def update_state_v2(self, all_nodes, ei_src, bs, src, pos_dst, t, msg, n_id, last_update, z ):
        used = ei_src.unique()
        all = all_nodes.unique()
        not_used = all[~isin(all, used)]
        is_src = not_used<bs
        not_used = not_used % bs

        s_store_indx = not_used[is_src]
        s_store_src = src[s_store_indx]
        s_store_dst = pos_dst[s_store_indx]
        s_store_t = t[s_store_indx]
        s_store_msg = msg[s_store_indx]

        d_store_indx = not_used[~is_src]
        d_store_src = pos_dst[d_store_indx]
        d_store_dst = src[d_store_indx]
        d_store_t = t[d_store_indx]
        d_store_msg = msg[d_store_indx]

        unique_nid, inverse = torch.unique(n_id, return_inverse=True)
        max_val, argmax_idx = scatter_max(last_update, inverse, dim=0)
        valid_mask = max_val > 0
        final_indices = argmax_idx[valid_mask]
        m_last_update = last_update[final_indices]
        m_nid = n_id[final_indices]
        m_memory = z[final_indices]

        self.memory[m_nid] = m_memory
        self.last_update[m_nid] = m_last_update

        
        self._update_msg_store(s_store_src, s_store_dst, s_store_t, s_store_msg, self.msg_s_store)
        self._update_msg_store(d_store_src, d_store_dst, d_store_t, d_store_msg, self.msg_d_store)

        # if not self.training:
        #     self._update_memory(n_id)





    # def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
    #     """Updates the memory with newly encountered interactions
    #     :obj:`(src, dst, t, raw_msg)`."""
    #     n_id = torch.cat([src, dst]).unique()

    #     if self.training:
    #         self._update_memory(n_id)
    #         self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
    #         self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
    #     else:
    #         self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
    #         self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
    #         self._update_memory(n_id)

    def _reset_message_store(self):
        i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        # Message store format: (src, dst, t, msg)
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def _update_memory(self, n_id: Tensor):
        memory, last_update = self._get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    # def _intra_batch_compute_msg(self, all_n_id, b_edge_index, b_isrc, b_raw_msg, b_t, last_update, bm, msg_module):
    #     msrc_s = b_edge_index[1][b_isrc]
    #     # breakpoint()
    #     src_s = all_n_id[msrc_s]
    #     dst_s = all_n_id[b_edge_index[0][b_isrc]]
    #     raw_msg_s = b_raw_msg[b_isrc]
    #     t_s = b_t[b_isrc]
    #     t_rel_s = t_s - last_update[self._assoc[src_s]]
    #     t_enc_s = self.time_enc(t_rel_s.to(raw_msg_s.dtype))
    #     msg_s = msg_module(bm[self._assoc[src_s]], bm[self._assoc[dst_s]], raw_msg_s, t_enc_s)

    #     return msg_s, t_s, src_s, dst_s, msrc_s
    
    def _intra_batch_compute_msg_v2(self, all_n_id, b_edge_index, b_isrc, b_raw_msg, b_t, last_update, bm, msg_module):
        msrc_s = b_edge_index[1][b_isrc]
        msrc_d = b_edge_index[0][b_isrc]
        src_s = all_n_id[msrc_s]
        dst_s = all_n_id[b_edge_index[0][b_isrc]]
        raw_msg_s = b_raw_msg[b_isrc]
        t_s = b_t[b_isrc]
        t_rel_s = t_s - last_update[self._assoc[src_s]]
        t_enc_s = self.time_enc(t_rel_s.to(raw_msg_s.dtype))
        msg_s = msg_module(bm[msrc_s], bm[msrc_d], raw_msg_s, t_enc_s)

        return msg_s, t_s, src_s, dst_s, msrc_s



    def _apply_intra_batch_info_v2(self, all_n_id, old_mem, last_update, b_edge_index, b_t, b_raw_msg, b_isrc, init_mem = None, delivery_addr = None):
        msg_s, t_s, src_s, dst_s, msrc_s  = self._intra_batch_compute_msg_v2(all_n_id, b_edge_index, b_isrc, b_raw_msg, b_t, last_update, old_mem, self.msg_s_module)
        msg_d, t_d, src_d, dst_d, msrc_d  = self._intra_batch_compute_msg_v2(all_n_id, b_edge_index, ~b_isrc, b_raw_msg, b_t, last_update, old_mem, self.msg_d_module)

        # Aggregate messages.
        if delivery_addr is not None:
            msrc_s = delivery_addr[b_isrc]
            msrc_d = delivery_addr[~b_isrc]
        
        idx = torch.cat([msrc_s, msrc_d], dim=0).long()
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        # breakpoint()
        aggr = self.aggr_module(msg, idx, t, all_n_id.size(0))
        # breakpoint()

        # Get local copy of updated memory.
        if init_mem is None:
            memory = self.memory_updater(aggr, old_mem)
        else:
            memory = self.memory_updater(aggr, init_mem)
        dim_size = memory.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce="max")
        # breakpoint()
        return memory, last_update


    # def _apply_intra_batch_info(self, all_n_id, bm, last_update, b_edge_index, b_t, b_raw_msg, b_isrc):
    #     # breakpoint()
    #     # print(self._assoc[all_n_id])
    #     # print(all_n_id.max())
    #     # print(self._assoc[all_n_id].max())
    #     # print(bm.shape)

    #     old_mem = bm[self._assoc[all_n_id]]

    #     # breakpoint()

    #     msg_s, t_s, src_s, dst_s, msrc_s  = self._intra_batch_compute_msg(all_n_id, b_edge_index, b_isrc, b_raw_msg, b_t, last_update, bm, self.msg_s_module)
    #     msg_d, t_d, src_d, dst_d, msrc_d  = self._intra_batch_compute_msg(all_n_id, b_edge_index, ~b_isrc, b_raw_msg, b_t, last_update, bm, self.msg_d_module)

    #     # Aggregate messages.
    #     idx = torch.cat([msrc_s, msrc_d], dim=0).long()
    #     msg = torch.cat([msg_s, msg_d], dim=0)
    #     t = torch.cat([t_s, t_d], dim=0)
    #     # breakpoint()
    #     aggr = self.aggr_module(msg, idx, t, all_n_id.size(0))
    #     # breakpoint()

    #     # Get local copy of updated memory.
    #     memory = self.memory_updater(aggr, old_mem)
    #     dim_size = memory.size(0)
    #     last_update = scatter(t, idx, 0, dim_size, reduce="max")
    #     # breakpoint()

    #     # # Get local copy of updated `last_update`.
    #     # dim_size = self.last_update.size(0)
    #     # last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id]
    #     # breakpoint()
    #     return memory, last_update





    def _get_updated_memory(self, all_n_id: Tensor) -> Tuple[Tensor, Tensor]:
        n_id  = all_n_id.unique()
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, dst_s = self._compute_msg(
            n_id, self.msg_s_store, self.msg_s_module
        )

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, dst_d = self._compute_msg(
            n_id, self.msg_d_store, self.msg_d_module
        )

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # Get local copy of updated memory.
        memory = self.memory_updater(aggr, self.memory[n_id])

        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id]

        return memory, last_update

    def _update_msg_store(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        raw_msg: Tensor,
        msg_store: TGNMessageStoreType,
    ):
        
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        # breakpoint()
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])
        # for node, s, d, ts, msg in zip(src.tolist(), src, dst, t, raw_msg):
        #     msg_store[node] = (s, d, ts, msg)

        # # breakpoint()
        # # return {
        # #     src[i].item(): (src[i], dst[i], t[i], raw_msg[i])
        # #     for i in range(src.size(0))
        # # }
        # src_cpu = src.cpu()  # avoid repeated .item() GPU→CPU syncs
        # for i in range(src.size(0)):
        #     node_id = src_cpu[i].item()
        #     msg_store[node_id] = (src[i], dst[i], t[i], raw_msg[i])

    def _compute_msg(
        self, n_id: Tensor, msg_store: TGNMessageStoreType, msg_module: Callable
    ):
        data = [msg_store[i] for i in n_id.tolist()]
        # breakpoint()
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)
        # breakpoint()
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)

        return msg, t, src, dst

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            # breakpoint()
            self._update_memory(torch.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
        super().train(mode)
