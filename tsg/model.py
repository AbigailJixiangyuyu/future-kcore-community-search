import torch
from torch import nn
import torch.nn.functional as F


class HopTransformer(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_heads, num_layers, dropout, max_k=2):
        super().__init__()
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.hop_pe = nn.Parameter(torch.zeros(max_k + 1, hidden_dim))
        nn.init.normal_(self.hop_pe, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens):
        h = self.proj(tokens)
        h = h + self.hop_pe.unsqueeze(0)
        h = self.encoder(h)
        h = self.norm(h)
        return h[:, 0, :]


class TemporalTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, dropout, max_len=20):
        super().__init__()
        self.time_pe = nn.Parameter(torch.zeros(max_len, hidden_dim))
        nn.init.normal_(self.time_pe, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z_seq):
        L = z_seq.size(1)
        h = z_seq + self.time_pe[:L].unsqueeze(0)
        h = self.encoder(h)
        h = self.norm(h)
        return h[:, -1, :]


class EdgeDecoder(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim, L, dropout):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim * L, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        pair_in = 3 * hidden_dim + hidden_dim // 2
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, edge_hist):
        h_sum = h.unsqueeze(0) + h.unsqueeze(1)
        h_diff = torch.abs(h.unsqueeze(0) - h.unsqueeze(1))
        h_prod = h.unsqueeze(0) * h.unsqueeze(1)
        edge_repr = self.edge_mlp(edge_hist)
        pair_feat = torch.cat([h_sum, h_diff, h_prod, edge_repr], dim=-1)
        scores = self.pair_mlp(pair_feat).squeeze(-1)
        N = scores.size(0)
        idx = torch.triu_indices(N, N, offset=1, device=scores.device)
        upper_scores = scores[idx[0], idx[1]]
        return upper_scores, idx


class TSGModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hop_transformer = HopTransformer(
            feat_dim=config.node_feat_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_hop_layers,
            dropout=config.dropout,
            max_k=config.K,
        )
        self.temporal_transformer = TemporalTransformer(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_temporal_layers,
            dropout=config.dropout,
            max_len=20,
        )
        self.edge_decoder = EdgeDecoder(
            hidden_dim=config.hidden_dim,
            edge_feat_dim=config.edge_feat_dim,
            L=config.L,
            dropout=config.dropout,
        )

    def forward(self, hop_tokens, edge_feat, node_mask=None):
        L, N, K1, feat_dim = hop_tokens.shape
        h = hop_tokens.reshape(L * N, K1, feat_dim)
        h = self.hop_transformer(h)
        h = h.reshape(L, N, -1)
        h = h.permute(1, 0, 2)
        h = self.temporal_transformer(h)
        ef = edge_feat.permute(1, 2, 0, 3).reshape(N, N, -1)
        scores, _ = self.edge_decoder(h, ef)
        return scores
