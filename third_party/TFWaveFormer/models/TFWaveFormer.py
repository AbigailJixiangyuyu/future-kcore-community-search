import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import math

from .modules import TimeEncoder
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.utils import NeighborSampler


class TFWaveFormer(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: "NeighborSampler",
                 time_feat_dim: int, channel_embedding_dim: int, num_layers: int = 2, num_heads: int = 8,
                 dropout: float = 0.1, max_input_sequence_length: int = 128, device: str = 'cpu'):
        """
        FreeDyG with Wavelet-domain Transformer enhancement
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param device: str, device
        """
        super(TFWaveFormer, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
        
        self.nif_feat_dim = self.channel_embedding_dim
        self.nif_encoder = NIFEncoder(nif_feat_dim=self.nif_feat_dim, device=self.device)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'nif': nn.Linear(in_features=self.nif_feat_dim, out_features=self.edge_feat_dim, bias=True)
        })
        self.reduce_layer = nn.Linear(4 * self.edge_feat_dim, self.edge_feat_dim)
       
        self.wavelet_transformers = nn.ModuleList([
            WaveletTransformerLayer(
                d_model=self.edge_feat_dim,
                nhead=self.num_heads,
                max_seq_len=max_input_sequence_length,
                dropout=dropout,
                device=device
            )
            for _ in range(self.num_layers)
        ])

        self.pos_encoder = PositionalEncoding(self.edge_feat_dim, max_input_sequence_length, device)
        
        self.final_attention = MultiModalAttention(self.edge_feat_dim, self.num_heads, dropout)
        self.output_norm = nn.LayerNorm(self.edge_feat_dim)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings with wavelet-domain transformer
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        src_nodes_neighbor_ids, src_nodes_edge_ids, src_nodes_neighbor_times = \
           self.neighbor_sampler.get_historical_neighbors(node_ids=src_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=self.max_input_sequence_length)

        dst_nodes_neighbor_ids, dst_nodes_edge_ids, dst_nodes_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=dst_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=self.max_input_sequence_length)

        src_nodes_nif_features, dst_nodes_nif_features = \
            self.nif_encoder(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                           src_nodes_neighbor_ids=src_nodes_neighbor_ids,
                           dst_nodes_neighbor_ids=dst_nodes_neighbor_ids)

        src_features = self._get_multimodal_features(
            node_interact_times, src_nodes_neighbor_ids, src_nodes_edge_ids, 
            src_nodes_neighbor_times, src_nodes_nif_features
        )
        
        dst_features = self._get_multimodal_features(
            node_interact_times, dst_nodes_neighbor_ids, dst_nodes_edge_ids, 
            dst_nodes_neighbor_times, dst_nodes_nif_features
        )

        src_features = self.pos_encoder(src_features)
        dst_features = self.pos_encoder(dst_features)

        for wavelet_transformer in self.wavelet_transformers:
            src_features = wavelet_transformer(src_features)
            dst_features = wavelet_transformer(dst_features)

        src_embeddings = self.final_attention(src_features)
        dst_embeddings = self.final_attention(dst_features)

        src_embeddings = self.output_norm(src_embeddings)
        dst_embeddings = self.output_norm(dst_embeddings)

        return src_embeddings, dst_embeddings

    def _get_multimodal_features(self, node_interact_times, nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times, nodes_nif_features):
        nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, 
                            nodes_neighbor_ids=nodes_neighbor_ids,
                            nodes_edge_ids=nodes_edge_ids, 
                            nodes_neighbor_times=nodes_neighbor_times, 
                            time_encoder=self.time_encoder)

        node_features = self.projection_layer['node'](nodes_neighbor_node_raw_features)
        edge_features = self.projection_layer['edge'](nodes_edge_raw_features)
        time_features = self.projection_layer['time'](nodes_neighbor_time_features)
        nif_features = self.projection_layer['nif'](nodes_nif_features)

        combined_features = torch.cat([node_features, edge_features, time_features, nif_features], dim=-1)
        combined_features = self.reduce_layer(combined_features)

        return combined_features

    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge and time features
        """
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]
        nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device))
        
        nodes_neighbor_time_features[torch.from_numpy(nodes_neighbor_ids == 0)] = 0.0
        
        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features

    def set_neighbor_sampler(self, neighbor_sampler: "NeighborSampler"):
        """set neighbor sampler"""
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class WaveletTransformerLayer(nn.Module):
    
    def __init__(self, d_model: int, nhead: int, max_seq_len: int, dropout: float = 0.1, device: str = 'cpu'):
        super(WaveletTransformerLayer, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.max_seq_len = max_seq_len
        self.device = device
        
        self.wavelet_filter = WaveletFilter(d_model, max_seq_len, device=device)
        
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask=None, src_key_padding_mask=None):
        """
        :param x: Tensor, shape (batch_size, seq_len, d_model)
        :return: Tensor, shape (batch_size, seq_len, d_model)
        """
        wavelet_enhanced = self.wavelet_filter(x)
        x = self.norm1(x + wavelet_enhanced)
        
        # Older PyTorch attention expects (sequence, batch, features).
        sequence_first = x.transpose(0, 1)
        attn_output, _ = self.self_attn(sequence_first, sequence_first, sequence_first,
                                      attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        attn_output = attn_output.transpose(0, 1)
        x = self.norm2(x + self.dropout(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        
        return x


class WaveletFilter(nn.Module):
    
    def __init__(self, d_model: int, max_seq_len: int, device: str = 'cpu'):
        super(WaveletFilter, self).__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device
        
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2, groups=d_model)
            for k in [3, 5, 7, 9]  
        ])
        
        self.scale_weights = nn.Parameter(torch.ones(4, d_model) * 0.25)
        
        self.feature_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor, shape (batch_size, seq_len, d_model)
        :return: Tensor, shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        x_conv = x.transpose(1, 2)
        
        multi_scale_outputs = []
        for i, conv in enumerate(self.multi_scale_convs):
            scale_output = conv(x_conv)  # (batch_size, d_model, seq_len)
            multi_scale_outputs.append(scale_output)
        
        multi_scale_outputs = torch.stack(multi_scale_outputs, dim=0)  # (num_scales, batch_size, d_model, seq_len)
        
        scale_weights = torch.softmax(self.scale_weights, dim=0)  # (num_scales, d_model)
        scale_weights = scale_weights.view(4, 1, d_model, 1)  # (num_scales, 1, d_model, 1)
        
        weighted_output = (multi_scale_outputs * scale_weights).sum(dim=0)  # (batch_size, d_model, seq_len)
        
        weighted_output = weighted_output.transpose(1, 2)
        
        transformed = self.feature_transform(weighted_output)
        
        gate_weights = self.gate(torch.mean(x, dim=1, keepdim=True))
        transformed = transformed * gate_weights
        
        return transformed


class MultiModalAttention(nn.Module):
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super(MultiModalAttention, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        self.query_net = nn.Linear(d_model, d_model)
        
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor, shape (batch_size, seq_len, d_model)
        :return: Tensor, shape (batch_size, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        global_query = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1, d_model)
        global_query = self.query_net(global_query)
        
        attended_output, attention_weights = self.multihead_attn(
            global_query.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1)
        )  # (1, batch_size, d_model)
        attended_output = attended_output.transpose(0, 1)
        
        output = self.output_proj(attended_output.squeeze(1))  # (batch_size, d_model)
        
        return output


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, max_len: int, device: str = 'cpu'):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe.to(device))
        
    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor, shape (batch_size, seq_len, d_model)
        :return: Tensor, shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)


class NIFEncoder(nn.Module):

    def __init__(self, nif_feat_dim: int, device: str = 'cpu'):
        super(NIFEncoder, self).__init__()

        self.nif_feat_dim = nif_feat_dim
        self.device = device

        self.nif_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.nif_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.nif_feat_dim, out_features=self.nif_feat_dim))

    def count_nodes_appearances(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_nodes_neighbor_ids: np.ndarray, dst_nodes_neighbor_ids: np.ndarray):
        src_nodes_appearances, dst_nodes_appearances = [], []
        
        for i in range(len(src_node_ids)):
            src_node_id = src_node_ids[i]
            dst_node_id = dst_node_ids[i]
            src_node_neighbor_ids = src_nodes_neighbor_ids[i]
            dst_node_neighbor_ids = dst_nodes_neighbor_ids[i]

            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_node_neighbor_ids, return_inverse=True, return_counts=True)
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_node_neighbor_ids, return_inverse=True, return_counts=True)

            src_mapping_dict = dict(zip(src_unique_keys, src_counts))
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            if src_node_id in dst_mapping_dict:
                src_count_in_dst = dst_mapping_dict[src_node_id]
                src_mapping_dict[src_node_id] = src_count_in_dst
                dst_mapping_dict[src_node_id] = src_count_in_dst
            if dst_node_id in src_mapping_dict:
                dst_count_in_src = src_mapping_dict[dst_node_id]
                src_mapping_dict[dst_node_id] = dst_count_in_src
                dst_mapping_dict[dst_node_id] = dst_count_in_src

            src_node_neighbor_counts_in_dst = torch.tensor([dst_mapping_dict.get(neighbor_id, 0) for neighbor_id in src_node_neighbor_ids]).float().to(self.device)
            dst_node_neighbor_counts_in_src = torch.tensor([src_mapping_dict.get(neighbor_id, 0) for neighbor_id in dst_node_neighbor_ids]).float().to(self.device)

            src_nodes_appearances.append(torch.stack([torch.from_numpy(src_counts[src_inverse_indices]).float().to(self.device), src_node_neighbor_counts_in_dst], dim=1))
            dst_nodes_appearances.append(torch.stack([dst_node_neighbor_counts_in_src, torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(self.device)], dim=1))

        src_nodes_appearances = torch.stack(src_nodes_appearances, dim=0)
        dst_nodes_appearances = torch.stack(dst_nodes_appearances, dim=0)

        return src_nodes_appearances, dst_nodes_appearances

    def forward(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_nodes_neighbor_ids: np.ndarray, dst_nodes_neighbor_ids: np.ndarray):
        src_nodes_appearances, dst_nodes_appearances = self.count_nodes_appearances(src_node_ids=src_node_ids,dst_node_ids=dst_node_ids,src_nodes_neighbor_ids=src_nodes_neighbor_ids, dst_nodes_neighbor_ids=dst_nodes_neighbor_ids)

        src_nodes_nif_features = self.nif_encode_layer(src_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_nodes_nif_features = self.nif_encode_layer(dst_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        
        return src_nodes_nif_features, dst_nodes_nif_features
