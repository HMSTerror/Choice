"""The NewChoice train-only recent-lag neighbor construction, without other variants."""
from collections import Counter, defaultdict

import torch


def build_neighbors(sequence, targets, item_count, count=16, lags=3, self_prior=1.0):
    rows = defaultdict(Counter)
    for history, target in zip(sequence.tolist(), targets.tolist()):
        valid = [item for item in history if 0 <= item < item_count]
        for lag, source in enumerate(reversed(valid[-lags:]), 1):
            rows[source][target] += 1.0 / lag
    width = min(count, item_count)
    ids = torch.arange(item_count)[:, None].expand(-1, width).clone()
    weights = torch.zeros(item_count, width)
    weights[:, 0] = 1.0
    for source, counts in rows.items():
        counts[source] += self_prior
        selected = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:width]
        ids[source, :len(selected)] = torch.tensor([item for item, _ in selected])
        weights[source].zero_()
        counts = torch.tensor([value for _, value in selected], dtype=torch.float32)
        weights[source, :len(selected)] = counts / counts.sum()
    return ids, weights
