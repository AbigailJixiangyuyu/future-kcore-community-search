import argparse

import numpy as np
import torch

from tsg.config import V1Config
from tsg.features import compute_all_features
from tsg.dataset import make_datasets
from tsg.model import TSGModel
from tsg.train import train, evaluate_epoch
from tsg.evaluate import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="email-Eu-core-temporal")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--hidden-dim", default=64, type=int)
    parser.add_argument("--window", default=604800, type=int, help="snapshot window in seconds")
    parser.add_argument("--L", default=5, type=int, help="history window length")
    parser.add_argument("--K", default=2, type=int, help="max hop")
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = V1Config(
        dataset_name=args.dataset,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        window_sec=args.window,
        L=args.L,
        K=args.K,
        seed=args.seed,
    )

    print("=" * 60)
    print("TSG V1 — Temporal Structural Transformer Graph Forecasting")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Config: {config}")

    print("\n[1/4] Loading data & computing features...")
    node_feat, hop_tokens, adj_matrices, node_map, num_nodes = compute_all_features(config)
    print(f"  node_feat: {node_feat.shape}")
    print(f"  hop_tokens: {hop_tokens.shape}")
    print(f"  {len(adj_matrices)} adjacency matrices of {num_nodes}x{num_nodes}")

    print("\n[2/4] Building datasets...")
    train_dataset, test_dataset = make_datasets(hop_tokens, adj_matrices, node_feat, config)
    print(f"  Train windows: {len(train_dataset)}")
    print(f"  Test windows: {len(test_dataset)}")

    print("\n[3/4] Creating model...")
    model = TSGModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    print("\n[4/4] Training...")
    model = train(model, train_dataset, test_dataset, device, config)

    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    test_scores, test_labels = evaluate_epoch(model, test_dataset, device, config)
    test_metrics = compute_metrics(test_scores, test_labels)

    print(f"  ROC-AUC:       {test_metrics['auc']:.4f}")
    print(f"  AP:            {test_metrics['ap']:.4f}")
    print(f"  Precision@M:   {test_metrics['precision_at_m']:.4f}")
    print(f"  Recall@M:      {test_metrics['recall_at_m']:.4f}")
