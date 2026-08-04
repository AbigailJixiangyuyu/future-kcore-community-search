import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_epoch(model, dataset, optimizer, device, config):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for idx in range(len(dataset)):
        hop_tokens, edge_feat, labels = dataset[idx]
        hop_tokens = hop_tokens.to(device)
        edge_feat = edge_feat.to(device)
        labels = labels.to(device)

        scores = model(hop_tokens, edge_feat)

        num_pos = labels.sum().item()
        num_neg = labels.numel() - num_pos
        if num_pos > 0:
            pos_weight = torch.tensor([num_neg / num_pos], device=device)
        else:
            pos_weight = torch.tensor([1.0], device=device)

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(model, dataset, device, config):
    model.eval()
    all_scores = []
    all_labels = []

    for idx in range(len(dataset)):
        hop_tokens, edge_feat, labels = dataset[idx]
        hop_tokens = hop_tokens.to(device)
        edge_feat = edge_feat.to(device)

        scores = model(hop_tokens, edge_feat)
        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


def train(model, train_dataset, test_dataset, device, config):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    best_ap = 0.0
    best_state = None

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_dataset, optimizer, device, config)
        train_scores, train_labels = evaluate_epoch(model, train_dataset, device, config)
        test_scores, test_labels = evaluate_epoch(model, test_dataset, device, config)

        from tsg.evaluate import compute_metrics
        train_metrics = compute_metrics(train_scores, train_labels)
        test_metrics = compute_metrics(test_scores, test_labels)

        if test_metrics["ap"] > best_ap:
            best_ap = test_metrics["ap"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1:3d}/{config.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train AP: {train_metrics['ap']:.4f} AUC: {train_metrics['auc']:.4f} | "
              f"Test AP: {test_metrics['ap']:.4f} AUC: {test_metrics['auc']:.4f} | "
              f"Test P@M: {test_metrics['precision_at_m']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
