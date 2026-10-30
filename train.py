"""Train the proposer, freeze candidates, train score entropy, and evaluate once."""
from contextlib import contextmanager
from copy import deepcopy
from itertools import count
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import SPLITS, loader, load_split, metadata, prepared_root, write_json
from diffusion import ChoiceProcess, random_partition
from models import Denoiser, SASRec
from neighbors import build_neighbors


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def training_epochs(limit):
    """Zero leaves the horizon open; validation patience ends the training loop."""
    if limit < 0:
        raise ValueError("Epoch limits must be nonnegative")
    return count(1) if limit == 0 else range(1, limit + 1)


def append_log(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def ranking_totals(scores, target):
    ranked = scores.topk(min(10, scores.shape[1]), -1).indices
    matches = ranked == target[:, None]
    rank = matches.to(torch.int64).argmax(-1) + 1
    found = matches.any(-1)
    result = {}
    for cutoff in (5, 10):
        hit = found & (rank <= cutoff)
        result[f"hr@{cutoff}"] = float(hit.sum())
        result[f"ndcg@{cutoff}"] = float((hit / torch.log2(rank.float() + 1)).sum())
    return result


def better(metrics, previous):
    return previous is None or metrics["ndcg@10"] > previous["ndcg@10"]


@torch.no_grad()
def evaluate_proposer(model, values, batch_size, device):
    model.eval()
    totals = {}
    for sequence, lengths, target in loader(values, batch_size):
        scores = model(sequence.to(device), lengths.to(device))
        for key, value in ranking_totals(scores, target.to(device)).items():
            totals[key] = totals.get(key, 0.) + value
    return {key: value / len(values["target"]) for key, value in totals.items()}


def train_proposer(output, config, device):
    started = time.perf_counter()
    seed_all(config["seed"])
    n = metadata(output)["item_count"]
    settings = config["sasrec"]
    training, validation = load_split(output, "train"), load_split(output, "val")
    batches = loader(training, settings["batch_size"], shuffle=True, seed=config["seed"])
    model = SASRec(n, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=0.)
    checkpoint = Path(output) / "sasrec_best.pt"
    if checkpoint.exists():
        raise FileExistsError(f"Proposer already exists: {checkpoint}; choose a new output")
    best, best_epoch, best_seconds = None, 0, 0.
    for epoch in training_epochs(settings["epochs"]):
        model.train()
        total, examples = 0., 0
        for sequence, lengths, target in batches:
            sequence, lengths, target = sequence.to(device), lengths.to(device), target.to(device)
            state = model.encode(sequence, lengths)
            negative = torch.randint(n - 1, (len(target), settings["negatives"]), device=device)
            negative = negative + (negative >= target[:, None])
            positive_score = (state * model.items(target)).sum(-1)
            negative_score = (state[:, None] * model.items(negative)).sum(-1)
            loss = F.softplus(-positive_score).mean() + F.softplus(negative_score).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite SASRec loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(target)
            examples += len(target)
        metrics = evaluate_proposer(model, validation, settings["batch_size"], device)
        if better(metrics, best):
            best, best_epoch = metrics, epoch
            best_seconds = time.perf_counter() - started
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation": metrics,
                        "config": config}, checkpoint)
        append_log(Path(output) / "sasrec_log.jsonl", {"stage": "sasrec", "epoch": epoch,
                   "loss": total / examples, "validation": metrics, "best_epoch": best_epoch})
        if epoch - best_epoch >= settings["patience"]:
            break
    write_json(Path(output) / "sasrec_training.json", {"epochs_completed": epoch,
               "best_epoch": best_epoch, "best_validation": best, "selection_split": "val",
               "epoch_limit": settings["epochs"] or None,
               "stop_reason": "early_stopping" if epoch - best_epoch >= settings["patience"] else "epoch_limit",
               "selected_checkpoint_seconds": best_seconds,
               "elapsed_seconds": time.perf_counter() - started})


@torch.no_grad()
def build_artifacts(output, config, device):
    n = metadata(output)["item_count"]
    state = torch.load(Path(output) / "sasrec_best.pt", map_location=device, weights_only=True)
    model = SASRec(n, config).to(device)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    candidates = {}
    k = config["sasrec"]["candidate_count"]
    if not 0 < k < n - config["history_length"]:
        raise ValueError("candidate_count must leave noncandidates after history exclusion")
    for split in SPLITS:
        values = load_split(output, split)
        rows = []
        # Targets remain on CPU and never enter proposer scoring or masking.
        for sequence, lengths, _ in loader(values, config["training"]["eval_batch_size"]):
            sequence, lengths = sequence.to(device), lengths.to(device)
            scores = model(sequence, lengths)
            blocked = torch.zeros(len(sequence), n + 1, device=device, dtype=torch.bool)
            blocked.scatter_(1, sequence, True)
            scores.masked_fill_(blocked[:, :n], -torch.inf)
            rows.append(scores.topk(k, -1).indices.cpu())
        candidates[split] = torch.cat(rows).to(torch.int32)
    torch.save(candidates, Path(output) / "candidates.pt")
    training = load_split(output, "train")
    destination = config["destination"]
    ids, weights = build_neighbors(training["seq"], training["target"], n,
                destination["neighbor_count"], destination["neighbor_lags"], destination["neighbor_self_prior"])
    counts = torch.from_numpy(np.load(prepared_root(output) / "prepared" / "train_target_counts.npy")).float()
    base = (counts + 1).sqrt()
    settings = config["diffusion"]
    artifact = {"base": base / base.sum(), "neighbor_ids": ids, "neighbor_weights": weights,
                "blocks": random_partition(n, settings["fine_size"], settings["coarse_size"],
                                           settings["partition_seed"])}
    torch.save(artifact, Path(output) / "structure.pt")
    write_json(Path(output) / "artifacts.json", {
        "candidate_count": k, "proposer_epoch": state["epoch"], "proposer_frozen": True,
        "candidate_score_magnitudes_used": False, "statistics_split": "train",
        "candidate_gate": destination.get("candidate_gate", "fixed_mass"),
        "candidate_mass": destination["candidate_mass"], "neighbor_mix": destination["neighbor_mix"],
        "base_count_definition": "training next-item targets",
        "partition": "fixed random permutation; contiguous nested blocks; no behavior clustering",
        "block_counts": [int(torch.unique(ids).numel()) for ids in artifact["blocks"]],
        "neighbor_count": ids.shape[1], "neighbor_lags": destination["neighbor_lags"],
        "neighbor_self_prior": destination["neighbor_self_prior"]})


def load_process(output, config, device):
    artifact = torch.load(config.get("structure_file", Path(output) / "structure.pt"), map_location="cpu", weights_only=True)
    if config.get("method") == "prefergrow":
        from prefergrow import PreferGrowProcess
        return PreferGrowProcess(artifact, config).to(device)
    return ChoiceProcess(artifact, config).to(device)


def candidate_file(output, config):
    return config.get("candidate_file", Path(output) / "candidates.pt")


def make_denoiser(output, config, process):
    from copy import deepcopy
    settings = deepcopy(config)
    settings["catalog_count"] = metadata(output)["item_count"]
    return Denoiser(process.item_count, settings)


@contextmanager
def evaluation_rng(seed, split, stream, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        payload = f"{seed}:{split}:{stream}".encode("utf-8")
        torch.manual_seed(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1))
        yield


@torch.no_grad()
def evaluate_diffusion(model, process, values, candidates, config, device, split, streams, score_transform=None):
    model.eval()
    results = []
    for stream in range(streams):
        totals, substeps, cap_count = {}, 0., 0
        with evaluation_rng(config["seed"], split, stream, device):
            for sequence, _, target, proposed in loader(values, config["training"]["eval_batch_size"],
                                                        candidates=candidates):
                scores, telemetry = process.sample_scores(model, sequence.to(device), proposed.to(device),
                                                         config["training"]["guidance_strength"])
                if scores.shape[1] != process.item_count or not torch.isfinite(scores).all():
                    raise FloatingPointError("Expected finite full-catalog ranking scores")
                scores = scores[:, :model.catalog_count]
                if score_transform is not None:
                    scores = score_transform(scores, sequence.to(device))
                for key, value in ranking_totals(scores, target.to(device)).items():
                    totals[key] = totals.get(key, 0.) + value
                substeps += telemetry["mean_reverse_substeps"] * len(target)
                cap_count += telemetry["exit_cap_activations"]
        results.append({**{key: value / len(values["target"]) for key, value in totals.items()},
                        "mean_reverse_substeps": substeps / len(values["target"]),
                        "exit_cap_activations": cap_count})
    metrics = {key: sum(row[key] for row in results) / streams for key in results[0]}
    return {**metrics, "examples": len(values["target"]), "streams": streams,
            "per_stream": results}


class EMA:
    def __init__(self, model, decay):
        self.model = deepcopy(model).eval().requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for shadow, current in zip(self.model.parameters(), model.parameters()):
            shadow.lerp_(current, 1 - decay)


def train_diffusion(output, config, device):
    seed_all(config["seed"])
    settings = config["training"]
    process = load_process(output, config, device)
    model = make_denoiser(output, config, process).to(device)
    ema = EMA(model, settings["ema_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=0.)
    candidates = torch.load(candidate_file(output, config), map_location="cpu", weights_only=True)
    training, validation = load_split(output, "train"), load_split(output, "val")
    batches = loader(training, settings["batch_size"], shuffle=True, seed=config["seed"],
                     candidates=candidates["train"])
    checkpoint = Path(output) / "choice_best.pt"
    if checkpoint.exists():
        raise FileExistsError(f"Diffusion checkpoint already exists: {checkpoint}; choose a new output")
    best, best_epoch, best_seconds, started = None, 0, 0., time.perf_counter()
    objective = ("score_entropy+guided_pairwise" if config["diffusion"].get("guided_negative_weight", 0.)
                 else "score_entropy")
    for epoch in training_epochs(settings["epochs"]):
        model.train()
        total, examples = 0., 0
        for sequence, _, target, proposed in batches:
            loss = process.loss(model, sequence.to(device), target.to(device), proposed.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite score-entropy loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            ema.update(model)
            total += float(loss.detach()) * len(target)
            examples += len(target)
        metrics = None
        if epoch % settings["validation_interval"] == 0 or epoch == settings["epochs"]:
            metrics = evaluate_diffusion(ema.model, process, validation, candidates["val"], config,
                                         device, "val", settings["validation_streams"])
            if better(metrics, best):
                best, best_epoch = metrics, epoch
                best_seconds = time.perf_counter() - started
                torch.save({"model": ema.model.state_dict(), "epoch": epoch, "validation": metrics,
                            "config": config, "objective": objective, "ema_updates": ema.updates},
                           checkpoint)
        append_log(Path(output) / "choice_log.jsonl", {"stage": "choice", "epoch": epoch,
                   "score_entropy": total / examples, "validation": metrics, "best_epoch": best_epoch})
        if metrics is not None and epoch - best_epoch >= settings["patience"]:
            break
    write_json(Path(output) / "training.json", {"epochs_completed": epoch, "best_epoch": best_epoch,
               "best_validation": best, "selection_split": "val", "objective": objective,
               "epoch_limit": settings["epochs"] or None,
               "stop_reason": "early_stopping" if epoch - best_epoch >= settings["patience"] else "epoch_limit",
               "selected_checkpoint_seconds": best_seconds,
               "elapsed_seconds": time.perf_counter() - started})


def final_evaluation(output, config, device):
    process = load_process(output, config, device)
    state = torch.load(Path(output) / "choice_best.pt", map_location=device, weights_only=True)
    model = make_denoiser(output, config, process).to(device)
    model.load_state_dict(state["model"])
    candidates = torch.load(candidate_file(output, config), map_location="cpu", weights_only=True)
    result = evaluate_diffusion(model, process, load_split(output, "test"), candidates["test"], config,
                                device, "test", config["training"]["test_streams"])
    report = {"dataset": config["dataset"], "seed": config["seed"], "smoke": config["smoke"],
              "best_epoch": state["epoch"], "selection_split": "val", "objective": state["objective"],
              "neighbor_mix": config["destination"]["neighbor_mix"], "test": result}
    write_json(Path(output) / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report
