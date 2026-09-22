"""Shared chronological boundaries for snapshot community comparisons."""


def test_start_t(snapshot_count):
    """First held-out current snapshot; its target is the following snapshot."""
    if snapshot_count < 4:
        raise ValueError("at least four snapshots are required for train/val/test")
    return int(snapshot_count * 0.7)


def training_boundaries(snapshot_count):
    """Inclusive train and validation target indices, respectively."""
    test_start = test_start_t(snapshot_count)
    train_end = int(snapshot_count * 0.55) - 1
    if not 1 <= train_end < test_start < snapshot_count - 1:
        raise ValueError("not enough snapshots for nonempty train/val/test splits")
    return train_end, test_start
