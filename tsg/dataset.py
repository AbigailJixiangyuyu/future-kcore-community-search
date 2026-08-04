import numpy as np
import torch
from torch.utils.data import Dataset


class GraphWindowDataset(Dataset):
    def __init__(self, hop_tokens, adj_matrices, node_feat, config, indices=None):
        self.hop_tokens = hop_tokens
        self.adj_matrices = adj_matrices
        self.node_feat = node_feat
        self.config = config
        self.L = config.L
        self.N = hop_tokens.shape[1]
        self.T = hop_tokens.shape[0]
        self.num_pairs = self.N * (self.N - 1) // 2

        self.triu_idx = np.triu_indices(self.N, k=1)

        if indices is not None:
            self.window_indices = indices
        else:
            all_indices = list(range(self.T - self.L))
            split = int(len(all_indices) * config.split_ratio)
            self.window_indices = all_indices[:split]

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        w = self.window_indices[idx]

        ht = torch.from_numpy(self.hop_tokens[w:w + self.L].copy())

        edge_feat = self._compute_edge_features(w)

        target_adj = self.adj_matrices[w + self.L]
        labels = torch.from_numpy(target_adj[self.triu_idx].astype(np.float32))

        return ht, edge_feat, labels

    def _compute_edge_features(self, w):
        L = self.L
        N = self.N
        edge_feat = np.zeros((L, N, N, self.config.edge_feat_dim), dtype=np.float32)

        for t_offset in range(L):
            t = w + t_offset
            adj = self.adj_matrices[t]
            nf = self.node_feat[t]

            deg = adj.sum(axis=1)
            core = nf[:, 1]

            edge_feat[t_offset, :, :, 0] = adj

            cn = adj @ adj.T
            np.fill_diagonal(cn, 0)
            edge_feat[t_offset, :, :, 1] = cn

            edge_feat[t_offset, :, :, 2] = np.abs(deg[:, None] - deg[None, :])

            edge_feat[t_offset, :, :, 3] = np.abs(core[:, None] - core[None, :])

            edge_feat[t_offset, :, :, 4] = np.minimum(core[:, None], core[None, :])

            edge_feat[t_offset, :, :, 5] = deg[:, None] * deg[None, :]

        return torch.from_numpy(edge_feat)


def make_datasets(hop_tokens, adj_matrices, node_feat, config):
    T = hop_tokens.shape[0]
    L = config.L
    all_indices = list(range(T - L))
    split = int(len(all_indices) * config.split_ratio)

    train_indices = all_indices[:split]
    test_indices = all_indices[split:]

    train_dataset = GraphWindowDataset(hop_tokens, adj_matrices, node_feat, config, indices=train_indices)
    test_dataset = GraphWindowDataset(hop_tokens, adj_matrices, node_feat, config, indices=test_indices)

    return train_dataset, test_dataset
