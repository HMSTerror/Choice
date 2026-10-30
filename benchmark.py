"""Measure exact structured kernels and checkpoint inference on the current host."""
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from data import load_split, metadata, write_json
from diffusion import ChoiceProcess, random_partition
from train import candidate_file, evaluation_rng, load_process, make_denoiser
from models import SASRec
from checkpoint_policy import cleanup_completed


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def latency(function, device, warmup, repeats):
    for _ in range(warmup):
        function()
    synchronize(device)
    times = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        function()
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)), "p95_ms": float(np.quantile(times, .95)), "repeats": repeats}


def repeated_latency(function, device, warmup, repeats, runs):
    results = [latency(function, device, warmup, repeats) for _ in range(runs)]
    return {"mean_ms": float(np.mean([row["mean_ms"] for row in results])),
            "runs": runs, "per_run": results}


def synthetic_process(n, device):
    generator = torch.Generator().manual_seed(100)
    base = torch.rand(n, generator=generator) + .1
    base /= base.sum()
    artifact = {"base": base, "blocks": random_partition(n, 64, 1024, 100),
                "neighbor_ids": torch.arange(n)[:, None], "neighbor_weights": torch.ones(n, 1)}
    config = {"destination": {"candidate_mass": .02, "neighbor_mix": 0., "history_decay": .2},
              "diffusion": {"reset_rate": .03, "level_rates": [.4, .1, -math.log(.1) - .5]}}
    process = ChoiceProcess(artifact, config).to(device)
    history = torch.tensor([[1, 2, 3]], device=device)
    candidates = torch.arange(min(100, n - 1), device=device)[None]
    return process, process.context(history, candidates)


@torch.no_grad()
def operator_benchmark(settings, device, smoke=False):
    sizes = [32, 64] if smoke else settings["dense_sizes"]
    large = 256 if smoke else settings["large_size"]
    warmup, repeats = (0, 1) if smoke else (settings["warmup"], settings["repeats"])
    runs = 1 if smoke else settings["runs"]
    rows = []
    for n in sizes:
        process, context = synthetic_process(n, device)
        ids = torch.arange(n, device=device)
        # Construct the row generator with the same rho, blocks and reset.
        q = process.reset_rate * process.base.expand(n, -1).clone()
        for level, rate in enumerate(process.rates):
            members = process.blocks[level, :, None] == process.blocks[level, None, :]
            q += rate * context.rho * members / context.block_mass[level].T
        q[ids, ids] -= sum(process.rates) + process.reset_rate
        source = torch.tensor([0], device=device)
        sigma = torch.tensor([1.], device=device)
        structured = lambda: process.forward_probability(source, sigma, context)
        dense = lambda: torch.matrix_exp(q)[0:1]
        reverse = lambda: process.incoming_rates(source, context)
        reverse_error = float((reverse() - q[:, 0][None].clone().scatter_(1, source[:, None], 0.)).abs().max())
        error = float((dense() - structured()).abs().max())
        row = {"items": n, "dtype": str(q.dtype), "max_abs_error": error,
               "reverse_max_abs_error": reverse_error,
               "structured": repeated_latency(structured, device, warmup, repeats, runs),
               "dense": repeated_latency(dense, device, warmup, repeats, runs),
               "reverse_operator": repeated_latency(reverse, device, warmup, repeats, runs)}
        row["speedup"] = row["dense"]["mean_ms"] / row["structured"]["mean_ms"]
        rows.append(row)
        print(f"[operator benchmark] N={n} error={error:.3g} speedup={row['speedup']:.2f}", flush=True)
        del q
    process, context = synthetic_process(large, device)
    source, sigma = torch.tensor([0], device=device), torch.tensor([1.], device=device)
    probability = process.forward_probability(source, sigma, context)
    tensors = list(process.buffers()) + [context.rho, *context.block_mass]
    large_result = {"items": large, "dtype": str(process.base.dtype), "forward": repeated_latency(lambda: process.forward_probability(source, sigma, context), device, warmup, repeats, runs),
                    "reverse_operator": repeated_latency(lambda: process.incoming_rates(source, context), device, warmup, repeats, runs),
                    "storage_bytes": sum(t.numel() * t.element_size() for t in tensors),
                    "row_sum_error": float((probability.sum(-1) - 1).abs().max()),
                    "finite": bool(torch.isfinite(probability).all())}
    # Check several starting states and noise levels, without dense N x N storage.
    failures, maximum = 0, large_result["row_sum_error"]
    for start in torch.linspace(0, large - 1, 16).long().tolist():
        for noise in (.001, 1., 10.):
            p = process.forward_probability(torch.tensor([start], device=device), torch.tensor([noise], device=device), context)
            error = float((p.sum(-1) - 1).abs().max())
            maximum = max(maximum, error)
            failures += int(not bool(torch.isfinite(p).all()) or bool((p < 0).any()) or error > 1e-4)
    large_result.update(row_sum_error=maximum, operator_failures=failures, checked_rows=48)
    return {"dense_comparison": rows, "large_catalog": large_result}


@torch.no_grad()
def end_to_end(directory, device, settings, smoke):
    config = json.loads((directory / "resolved_config.json").read_text(encoding="utf-8"))
    process = load_process(directory, config, device)
    model = make_denoiser(directory, config, process).to(device).eval()
    state = torch.load(directory / "choice_best.pt", weights_only=True, map_location=device)
    model.load_state_dict(state["model"])
    del state
    values = load_split(directory, "test")
    source = Path(candidate_file(directory, config)).parent
    proposer = None
    if config["method"] == "choicegrow":
        proposer = SASRec(metadata(directory)["item_count"], config).to(device).eval()
        proposer.load_state_dict(torch.load(source / "sasrec_best.pt", weights_only=True, map_location=device)["model"])
    n = metadata(directory)["item_count"]
    def query(index):
        # CPU query retrieval, transfer, online proposer, generation, ranking.
        sequence = values["seq"][index:index + 1].to(device)
        lengths = values["length"][index:index + 1].to(device)
        if proposer is not None:
            scores = proposer(sequence, lengths)
            blocked = torch.zeros(1, n + 1, device=device, dtype=torch.bool).scatter_(1, sequence, True)
            scores.masked_fill_(blocked[:, :n], -torch.inf)
            proposed = scores.topk(config["sasrec"]["candidate_count"], -1).indices
        else:
            proposed = torch.empty(1, 0, dtype=torch.long, device=device)
        scores, telemetry = process.sample_scores(model, sequence, proposed, config["training"]["guidance_strength"])
        ranked = scores[:, :n].topk(10, -1).indices.cpu()
        return telemetry
    warmup = 1 if smoke else settings["warmup"]
    count = len(values["target"])
    with evaluation_rng(config["seed"], "warmup", 0, device):
        for i in range(warmup):
            query(i % count)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times, substeps, caps = [], 0., 0
    with evaluation_rng(config["seed"], "test", 0, device):
        for index in range(count):
            synchronize(device)
            started = time.perf_counter()
            telemetry = query(index)
            synchronize(device)
            times.append((time.perf_counter() - started) * 1000)
            substeps += telemetry["mean_reverse_substeps"]
            caps += telemetry["exit_cap_activations"]
            if (index + 1) % 1000 == 0:
                print(f"[end-to-end] {config['method']} {index + 1}/{count}", flush=True)
    training = json.loads((directory / "training.json").read_text())
    measured = {"batch_size": 1, "queries": count, "catalog_items": n,
                "median_ms": float(np.median(times)), "p95_ms": float(np.quantile(times, .95)),
                "queries_per_second": count / (sum(times) / 1000),
                "mean_reverse_substeps": substeps / count, "exit_cap_activations": caps,
                "memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "training_seconds": training["selected_checkpoint_seconds"],
                "training_gpu_hours": training["selected_checkpoint_seconds"] / 3600 if device.type == "cuda" else None,
                "training_total_seconds": training["elapsed_seconds"], "online_proposer": proposer is not None,
                "timed_path": "CPU query to full-catalog top-10 returned to CPU",
                "query_latency_ms": times}
    proposer_summary = source / "sasrec_training.json"
    if proposer is not None:
        seconds = json.loads(proposer_summary.read_text())["selected_checkpoint_seconds"]
        measured.update(proposer_seconds=seconds, proposer_gpu_hours=seconds / 3600 if device.type == "cuda" else None,
                        proposer_seconds_per_query=seconds / count)
    return measured


def partition_statistics(root):
    rows = []
    for dataset in (root / "experiments").iterdir():
        if not dataset.is_dir():
            continue
        for kind in ("shuffle_partition", "real_partition"):
            groups = {}
            for path in (dataset / kind).glob("*/report.json"):
                config = json.loads((path.parent / "resolved_config.json").read_text())
                report = json.loads(path.read_text())
                groups.setdefault(config["partition_seed"], []).append(report["test"]["ndcg@10"])
            if len(groups) < 2 or len(set(map(len, groups.values()))) != 1:
                continue
            n = len(next(iter(groups.values())))
            if n < 2:
                continue
            x = np.asarray(list(groups.values()))
            within = float(((x - x.mean(1, keepdims=True)) ** 2).sum() / (x.shape[0] * (n - 1)))
            between = float(n * x.mean(1).var(ddof=1))
            partition_var = max(0., (between - within) / n)
            if partition_var == 0:
                within = float(x.var(ddof=1))
            rows.append({"dataset": dataset.name, "scope": kind, "grand_mean": float(x.mean()),
                         "partition_sd_reml": math.sqrt(partition_var), "within_sd_reml": math.sqrt(within),
                         "total_sd_reml": math.sqrt(partition_var + within),
                         "partitions": len(groups), "training_seeds_per_partition": n})
    return rows


def run_benchmarks(root, suite, device, smoke):
    output = root / "efficiency"
    output.mkdir(exist_ok=True)
    config = {"device": str(device), "smoke": smoke, "settings": suite["benchmark"]}
    if device.type == "cuda":
        config["gpu"] = torch.cuda.get_device_name(device)
    if not (output / "operators.json").exists():
        write_json(output / "operators.json", {**config, **operator_benchmark(suite["benchmark"], device, smoke)})
    rows = []
    for dataset in (suite["benchmark"]["end_to_end_dataset"],):
        for arm in ("choicegrow", "prefergrow"):
            seed = suite["benchmark"]["end_to_end_seed"]
            directory = root / "experiments" / dataset / arm / f"seed_{seed}"
            if (directory / "report.json").exists():
                target = output / f"end_to_end_{dataset}_{arm}_{seed}.json"
                if not target.exists():
                    write_json(target, {"dataset": dataset, "method": arm, "seed": seed,
                                       **end_to_end(directory, device, suite["benchmark"], smoke)})
                rows.append(json.loads(target.read_text(encoding="utf-8")))
    write_json(output / "end_to_end.json", {**config, "runs": rows})
    write_json(root / "summary" / "partition_reml.json", partition_statistics(root))
    cleanup_completed(root, suite, suite["groups"], finished=True)
