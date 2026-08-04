from dataclasses import dataclass


@dataclass
class V1Config:
    dataset_name: str = "email-Eu-core-temporal"
    snapshot_cache_dir: str = "datasets/snapshot_cache"
    feature_cache_dir: str = "tsg/feature_cache"
    window_sec: int = 604800
    L: int = 5
    K: int = 2
    h_index_orders: int = 3
    node_feat_dim: int = 10
    edge_feat_dim: int = 6
    hidden_dim: int = 64
    num_heads: int = 4
    num_hop_layers: int = 1
    num_temporal_layers: int = 1
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50
    split_ratio: float = 0.7
    seed: int = 42
    grad_clip: float = 1.0
    num_pairs: int = 0

    @property
    def snapshot_path(self) -> str:
        return f"{self.snapshot_cache_dir}/snapshots_{self.dataset_name}_{self.window_sec}.pkl"
