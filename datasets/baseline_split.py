"""Shared chronological boundaries for snapshot community comparisons."""


LEGACY_SPLIT = "snapshot_55_15_30_v1"
CURRENT_SPLIT = "snapshot_70_15_15_v1"


def test_start_t(snapshot_count):
    """First held-out current snapshot; its target is the following snapshot."""
    if snapshot_count < 4:
        raise ValueError("at least four snapshots are required for train/val/test")
    return int(snapshot_count * 0.7)


def training_boundaries(snapshot_count, split_rule=LEGACY_SPLIT):
    """Inclusive train and validation target indices, respectively."""
    if split_rule == LEGACY_SPLIT:
        train_end, fit_end = int(snapshot_count * 0.55) - 1, test_start_t(snapshot_count)
    elif split_rule == CURRENT_SPLIT:
        train_end, fit_end = int(snapshot_count * 0.7) - 1, int(snapshot_count * 0.85)
    else:
        raise ValueError("unknown snapshot split rule: " + str(split_rule))
    if not 1 <= train_end < fit_end < snapshot_count - 1:
        raise ValueError("not enough snapshots for nonempty train/val/test splits")
    return train_end, fit_end
