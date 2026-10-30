"""Frozen, training-only candidate controls and nested scope construction."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from data import SPLITS, load_split, write_json
from diffusion import random_partition


def popularity_counts(training, item_count):
    values = torch.cat((training["seq"].flatten(), training["target"]))
    return torch.bincount(values[values < item_count], minlength=item_count)


def control_candidates(candidates, values, counts, seed, window=64):
    """Use training popularity only; never read a validation/test target."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.argsort(counts, descending=True, stable=True)
    mapping = torch.arange(len(counts))
    for group in order.split(window):
        if len(group) > 1:
            permutation = group[torch.randperm(len(group), generator=generator)]
            # A random cyclic bijection avoids duplicate candidate members.
            mapping[permutation] = torch.roll(permutation, 1)
    popularity, swapped = {}, {}
    for split in SPLITS:
        popularity[split] = mapping[candidates[split]]
        # Sort by the last observed item's training frequency, then swap rows locally.
        last = values[split]["seq"][:, -1]
        ordered = torch.argsort(counts[last], stable=True)
        donors = torch.arange(len(last))
        for group in ordered.split(window):
            if len(group) > 1:
                shuffled = group[torch.randperm(len(group), generator=generator)]
                donors[shuffled] = torch.roll(shuffled, 1)
        swapped[split] = candidates[split][donors].clone()
    return popularity, swapped


def directed_edges(training, item_count):
    edges = Counter()
    for history, target in zip(training["seq"].tolist(), training["target"].tolist()):
        chain = [v for v in history if v < item_count] + [target]
        edges.update(zip(chain[:-1], chain[1:]))
    return edges


def real_partition(training, item_count, fine_size, coarse_size, seed, restarts=3):
    """Directed Infomap fine modules and Markov-time consensus ordering.

    Community orders are packed into exactly the same 64/1024 capacities as
    Shuffle-Scope. Packing enforces nesting even when modules exceed a capacity.
    """
    from infomap import Infomap
    edges = directed_edges(training, item_count)
    memberships = []
    for time in (1., 2., 4., 8., 16.):
        best = None
        for restart in range(restarts):
            model = Infomap(f"--directed --two-level --silent --num-trials 1 --seed {seed + restart + 1} --markov-time {time}")
            for node in range(item_count):
                model.add_node(node)
            for (source, target), weight in sorted(edges.items()):
                model.add_link(source, target, float(weight))
            model.run()
            candidate = (float(model.codelength), {int(k): int(v) for k, v in model.get_modules().items()})
            if best is None or candidate[0] < best[0]:
                best = candidate
        memberships.append([best[1].get(i, item_count + i) for i in range(item_count)])
    fine = memberships[0]
    groups = defaultdict(list)
    for item, label in enumerate(fine):
        groups[label].append(item)
    # A consensus signature is the modal assignment at each Markov time.
    def signature(items):
        return tuple(sorted(Counter(level[i] for i in items).items(), key=lambda row: (-row[1], row[0]))[0][0]
                     for level in memberships[1:])
    rng = np.random.default_rng(seed)
    ordered = []
    for _, items in sorted(groups.items(), key=lambda pair: (signature(pair[1]), pair[0])):
        ordered.extend(rng.permutation(items).tolist())
    rank = torch.empty(item_count, dtype=torch.long)
    rank[torch.tensor(ordered)] = torch.arange(item_count)
    return torch.stack((rank // fine_size, rank // coarse_size, torch.zeros_like(rank)))


def build_controls(output, config):
    output = Path(output)
    original = torch.load(output / "candidates.pt", weights_only=True)
    structure = torch.load(output / "structure.pt", weights_only=True)
    values = {split: load_split(output, split) for split in SPLITS}
    counts = popularity_counts(values["train"], len(structure["base"]))
    popular, swap = control_candidates(original, values, counts, config["seed"])
    torch.save(popular, output / "candidates_popularity.pt")
    torch.save(swap, output / "candidates_swap.pt")
    write_json(output / "controls.json", {"statistics_split": "train", "match_window": 64,
               "candidate_controls": ["original", "popularity", "swap"],
               "target_used_for_validation_or_test_candidates": False})


def scope_artifact(source, destination, config, kind, seed):
    artifact = torch.load(Path(source) / "structure.pt", weights_only=True)
    n = len(artifact["base"])
    settings = config["diffusion"]
    if kind == "real":
        artifact["blocks"] = real_partition(load_split(source, "train"), n,
                        settings["fine_size"], settings["coarse_size"], seed,
                        config.get("suite", {}).get("infomap_restarts", 3))
    elif kind == "shuffle":
        artifact["blocks"] = random_partition(n, settings["fine_size"], settings["coarse_size"], seed)
    else:
        raise ValueError(kind)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, destination)
    return destination
