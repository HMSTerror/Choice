"""Linux-ready paper experiment workflow: prepare, candidates, experiments.

Note: .agents/notes/implemented/architecture/2026-10-30-paper-suite.md
"""
import argparse
from copy import deepcopy
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from data import ALIASES, SPLITS, load_split, metadata, prepare, write_json
from diffusion import ChoiceProcess
from experiment_artifacts import build_controls, scope_artifact
from models import SASRec
from baselines import BASELINES, test_baseline, train_baseline
from semantic import semantic_preprocessing
from train import (better, build_artifacts, candidate_file, evaluate_diffusion,
                   final_evaluation, load_process, make_denoiser, train_diffusion, train_proposer)


ROOT = Path(__file__).resolve().parent
FACTORIAL = ("shared_root", "user_root", "shared_three", "choicegrow")
MAIN = (*BASELINES, "prefergrow", "choicegrow")


def path_at_root(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def canonical(values):
    return [ALIASES[v.lower()] for v in values]


def config_for(base, dataset, seed, smoke=False):
    result = deepcopy(base)
    result.update(dataset=dataset, seed=seed, smoke=smoke)
    if smoke:
        result["sasrec"].update(epochs=1, batch_size=8)
        result["training"].update(epochs=1, batch_size=8, eval_batch_size=8,
                    validation_interval=1, validation_streams=1, test_streams=1)
    return result


def frozen_config(directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "resolved_config.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != config:
        raise ValueError(f"Settings changed at {directory}; choose a new --output")
    write_json(target, config)


def source_dir(root, dataset, seed):
    return root / "candidates" / dataset / f"seed_{seed}"


def preprocessing(root, base, suite, datasets, smoke):
    for dataset in datasets:
        directory = root / "data" / dataset
        config = config_for(base, dataset, suite["seeds"][0], smoke)
        frozen_config(directory, config)
        if not (directory / "prepared" / "metadata.json").exists():
            print(f"[prepare] {dataset} -> {directory}", flush=True)
            prepare(ROOT / "dataset", directory, dataset, config)
        else:
            print(f"[reuse prepared] {dataset}", flush=True)
        if not (directory / "semantic.pt").exists():
            print(f"[semantic preprocessing] {dataset}: PreferDiff whitening + DDSR OPQ", flush=True)
            semantic_preprocessing(ROOT / "dataset", directory, config, suite["semantic"])


def candidate_stage(root, base, suite, datasets, seeds, device, smoke):
    preprocessing(root, base, suite, datasets, smoke)
    for dataset in datasets:
        for seed in seeds:
            output = source_dir(root, dataset, seed)
            config = config_for(base, dataset, seed, smoke)
            frozen_config(output, config)
            write_json(output / "data_source.json", {"path": str(root / "data" / dataset)})
            if not (output / "sasrec_complete.json").exists():
                if (output / "sasrec_best.pt").exists():
                    # Restart only this interrupted SASRec fit, keeping a backup.
                    backup = output / ("interrupted_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
                    backup.mkdir()
                    for name in ("sasrec_best.pt", "sasrec_log.jsonl", "sasrec_training.json"):
                        if (output / name).exists():
                            shutil.move(str(output / name), str(backup / name))
                print(f"[SASRec] {dataset} seed={seed}", flush=True)
                train_proposer(output, config, device)
                write_json(output / "sasrec_complete.json", {"complete": True})
            if not (output / "report.json").exists():
                print(f"[test SASRec best] {dataset} seed={seed}", flush=True)
                test_baseline(output, {**config, "method": "sasrec", "candidate_file": str(output / "candidates.pt")}, device)
            if not (output / "controls.json").exists():
                print(f"[candidates] {dataset} seed={seed}", flush=True)
                build_artifacts(output, config, device)
                build_controls(output, config)
            state = torch.load(output / "sasrec_best.pt", weights_only=True, map_location="cpu")
            print(f"[candidates ready] {output} proposer_epoch={state['epoch']}", flush=True)


def arm_config(config, arm, suite):
    config = deepcopy(config)
    destination, diffusion = config["destination"], config["diffusion"]
    config["experiment"] = arm
    config["method"] = "choicegrow"
    if arm in BASELINES:
        config["method"] = arm
        config["baseline"] = deepcopy(suite["baselines"]["settings"])
        config["training"]["learning_rate"] = suite["baselines"]["learning_rates"][0]
        if config["smoke"]:
            config["baseline"]["preferdiff"].update(steps=8, sampling_steps=2)
            config["baseline"]["ddsr"].update(steps=4, skip=2)
        return config
    destination.update(mode="user", weighting="base", candidate_gate="fixed_mass")
    total = sum(diffusion["level_rates"])
    if arm.startswith("shared"):
        destination.update(mode="shared", neighbor_mix=0.)
    if arm in {"shared_root", "user_root", "proposal_root"}:
        diffusion["level_rates"] = [0., 0., total]
    if arm in {"proposal_root", "proposal_hierarchy"}:
        destination["weighting"] = "uniform"
    if arm == "dynamic_mass":
        destination["candidate_gate"] = "dynamic_mass"
    if arm == "uniform_rate":
        diffusion["level_rates"] = [total / 3] * 3
    if arm in {"prefergrow", "guided_negatives"}:
        config["method"] = "prefergrow"
        destination.update(mode="shared", neighbor_mix=0., candidate_gate="fixed_mass")
        config["prefergrow"] = {"variant": suite["prefergrow"]["variant"],
                                "gamma": suite["prefergrow"]["gamma"][config["dataset"]]}
        config["training"]["learning_rate"] = suite["prefergrow"]["learning_rate"][config["dataset"]]
        if arm == "guided_negatives":
            diffusion["guided_negative_weight"] = suite["guided_negative_weights"][1]
    return config


def attach_inputs(config, source, structure=None, candidate="original"):
    config = deepcopy(config)
    name = {"original": "candidates.pt", "popularity": "candidates_popularity.pt", "swap": "candidates_swap.pt"}[candidate]
    config["candidate_file"] = str(source / name)
    config["structure_file"] = str(structure or source / "structure.pt")
    return config


def experiment_config(base, suite, dataset, seed, arm, source, smoke, structure=None):
    config = config_for(base, dataset, seed, smoke)
    config = arm_config(config, arm, suite)
    candidate = "popularity" if arm == "popularity" else "swap" if arm == "history_swap" else "original"
    return attach_inputs(config, source, structure, candidate)


def train_run(directory, source, config, device, test=True):
    frozen_config(directory, config)
    write_json(directory / "data_source.json", {"path": str(source)})
    complete = directory / "training.json"
    baseline = config.get("method") in BASELINES
    if not complete.exists():
        checkpoint = "baseline_best.pt" if baseline else "choice_best.pt"
        if config.get("method") == "sasrec":
            shutil.copyfile(source / "sasrec_training.json", complete)
        elif (directory / checkpoint).exists():
            backup = directory / ("interrupted_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            backup.mkdir()
            for name in (checkpoint, "baseline_log.jsonl" if baseline else "choice_log.jsonl"):
                if (directory / name).exists():
                    shutil.move(str(directory / name), str(backup / name))
        if not complete.exists():
            print(f"[train] {directory}", flush=True)
            (train_baseline if baseline else train_diffusion)(directory, config, device)
    if test:
        if not (directory / "report.json").exists():
            print(f"[test immediately] {directory}", flush=True)
            if config.get("method") == "sasrec" and (source / "report.json").exists():
                shutil.copyfile(source / "report.json", directory / "report.json")
            else:
                (test_baseline if baseline else final_evaluation)(directory, config, device)
        return json.loads((directory / "report.json").read_text(encoding="utf-8"))
    return json.loads(complete.read_text(encoding="utf-8"))


def tune_baseline(root, base, suite, dataset, method, source, device, smoke, seed):
    """Independent baseline validation searches; never tests a search checkpoint."""
    target = root / "selected" / f"{dataset}_{method}.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))["config"]
    config = experiment_config(base, suite, dataset, seed, method, source, smoke)
    config["semantic_file"] = str(root / "data" / dataset / "semantic.pt")
    candidates = []
    enabled = suite["baselines"]["tuning"] and suite["tuning"]["enabled"]
    rates = suite["baselines"]["learning_rates"] if enabled and not smoke else [config["training"]["learning_rate"]]
    skips = suite["baselines"]["ddsr_skips"] if enabled and method == "ddsr" and not smoke else [config["baseline"]["ddsr"]["skip"]]
    for index, (rate, skip) in enumerate((r, s) for r in rates for s in skips):
        trial = deepcopy(config)
        trial["training"]["learning_rate"] = rate
        trial["baseline"]["ddsr"]["skip"] = skip
        if enabled:
            result = train_run(root / "baseline_tuning" / dataset / method / f"trial_{index:03d}", source, trial, device, test=False)
            candidates.append((result["best_validation"]["ndcg@10"], trial))
        else:
            candidates.append((0., trial))
    config = max(candidates, key=lambda row: row[0])[1]
    write_json(target, {"config": config, "selection_split": "val", "trials": len(candidates),
                       "test_evaluated_during_search": False})
    return config


def tune_dataset(root, base, suite, dataset, device, smoke, seed=None):
    seed = suite["seeds"][0] if seed is None else seed
    source = source_dir(root, dataset, seed)
    selected_path = root / "selected" / f"{dataset}.json"
    if selected_path.exists():
        return json.loads(selected_path.read_text(encoding="utf-8"))["config"]
    selected = config_for(base, dataset, seed, smoke)
    override = suite.get("dataset_overrides", {}).get(dataset, {})
    for section, values in override.items():
        selected[section].update(values)
    settings = suite["tuning"]
    if settings["enabled"] and not override:
        rng = np.random.default_rng(seed)
        rows = []
        trials = 1 if smoke else settings["trials"][dataset]
        for trial in range(trials):
            config = deepcopy(selected)
            low, high = settings["learning_rate"]
            config["training"]["learning_rate"] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            for key in ("candidate_mass", "reset_rate"):
                section = "destination" if key == "candidate_mass" else "diffusion"
                config[section][key] = float(rng.uniform(*settings[key]))
            root_share, fine_share = rng.uniform(*settings["root_share"]), rng.uniform(*settings["fine_share"])
            total = -math.log(.1)
            config["diffusion"]["level_rates"] = [total * (1 - root_share) * fine_share,
                      total * (1 - root_share) * (1 - fine_share), total * root_share]
            config = attach_inputs(arm_config(config, "choicegrow", suite), source)
            result = train_run(root / "tuning" / dataset / f"trial_{trial:03d}", source, config, device, test=False)
            rows.append((result["best_validation"]["ndcg@10"], trial, config))
        retrained = []
        for rank, (_, trial, config) in enumerate(sorted(rows, key=lambda row: row[0], reverse=True)[:settings["retrain_top"]]):
            config = deepcopy(config)
            config["seed"] = seed + 10000
            result = train_run(root / "tuning" / dataset / f"retrain_{rank}", source, config, device, test=False)
            retrained.append((result["best_validation"]["ndcg@10"], trial, config))
        _, selected_trial, selected = max(retrained, key=lambda row: row[0])
        selected["seed"] = seed
        write_json(root / "tuning" / dataset / "selection.json", {"selection_split": "val", "trial": selected_trial,
                   "trials": trials, "retrained": len(retrained), "test_evaluated_during_search": False})
    write_json(selected_path, {"selection_split": "val", "config": selected})
    return selected


def mean_hazard(source, config, device, split="val", diagnostics=False):
    process = load_process(source, config, device)
    values = load_split(source, split)
    candidates = torch.load(candidate_file(source, config), weights_only=True)[split]
    hazards, retention = [], []
    from data import loader
    for sequence, _, target, proposed in loader(values, config["training"]["eval_batch_size"], candidates=candidates):
        sequence, target, proposed = sequence.to(device), target.to(device), proposed.to(device)
        context = process.context(sequence, proposed)
        kappa = process.reset_rate * (1 - process.base[target])
        for level, rate in enumerate(process.rates):
            self_probability = context.rho.gather(1, target[:, None]).squeeze(1) / context.block_mass[level].gather(1, target[:, None]).squeeze(1)
            kappa = kappa + rate * (1 - self_probability)
        hazards.extend(kappa.tolist())
        if diagnostics:
            for time in torch.linspace(1., .001, config["diffusion"]["training_intervals"] + 1):
                sigma = config["diffusion"]["sigma_min"] * (config["diffusion"]["sigma_max"] / config["diffusion"]["sigma_min"]) ** float(time)
                p = process.forward_probability(target, torch.full_like(target, sigma, dtype=torch.float32), context)
                retention.extend(p.gather(1, target[:, None]).squeeze(1).tolist())
    result = {"mean": float(np.mean(hazards)), "p10": float(np.quantile(hazards, .1)), "p90": float(np.quantile(hazards, .9))}
    if diagnostics:
        def stats(values):
            return {"mean": float(np.mean(values)), "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9))}
        result.update(retention=stats(retention), replacement=stats(1 - np.asarray(retention)),
                      split=split, queries=len(values["target"]), time_points=config["diffusion"]["training_intervals"] + 1,
                      weighting="each query-time pair has equal weight")
    return result


def load_trained(directory, config, device):
    process = load_process(directory, config, device)
    model = make_denoiser(directory, config, process).to(device)
    model.load_state_dict(torch.load(directory / "choice_best.pt", weights_only=True, map_location=device)["model"])
    model.eval()
    return process, model


def initialization_control(directory, config, device):
    shared = config["experiment"] == "shared_three"
    target = directory / ("report_user_initialization.json" if shared else "report_shared_initialization.json")
    if target.exists():
        return
    cfg = deepcopy(config)
    cfg["diffusion"]["initialization"] = "matched" if shared else "shared_base"
    process, model = load_trained(directory, cfg, device)
    if shared:
        reference = deepcopy(config)
        reference["destination"].update(mode="user", neighbor_mix=config.get("suite_neighbor_mix", .02))
        reference["diffusion"]["level_rates"] = config["full_level_rates"]
        process.reference_process = load_process(directory, reference, device)
    candidates = torch.load(candidate_file(directory, cfg), weights_only=True)["test"]
    result = evaluate_diffusion(model, process, load_split(directory, "test"), candidates, cfg, device, "test", cfg["training"]["test_streams"])
    write_json(target, {"test": result, "initialization": "user_terminal" if shared else "shared_base", "training_unchanged": True})


def fusion_control(directory, source, config, suite, device):
    target = directory / "report_score_fusion.json"
    if target.exists():
        return
    process, model = load_trained(directory, config, device)
    proposer = SASRec(metadata(directory)["item_count"], config).to(device).eval()
    proposer.load_state_dict(torch.load(source / "sasrec_best.pt", weights_only=True, map_location=device)["model"])
    candidates = torch.load(candidate_file(directory, config), weights_only=True)
    def transform(weight):
        def apply(scores, sequence):
            lengths = (sequence < proposer.item_count).sum(-1)
            return scores + weight * proposer(sequence, lengths)
        return apply
    best, alpha = None, 0.
    for weight in suite["fusion_weights"]:
        metrics = evaluate_diffusion(model, process, load_split(directory, "val"), candidates["val"], config, device,
                    "val", config["training"]["validation_streams"], transform(weight))
        if better(metrics, best):
            best, alpha = metrics, weight
    test = evaluate_diffusion(model, process, load_split(directory, "test"), candidates["test"], config, device,
                    "test", config["training"]["test_streams"], transform(alpha))
    write_json(target, {"test": test, "fusion_weight": alpha, "selection_split": "val"})


@torch.no_grad()
def scope_diagnostics(directory, config, device):
    target = directory / "scope_diagnostics.json"
    if target.exists():
        return
    from data import loader
    process = load_process(directory, config, device)
    candidates = torch.load(candidate_file(directory, config), weights_only=True)["test"]
    totals = np.zeros((2, 3))
    examples = 0
    for sequence, _, _, proposed in loader(load_split(directory, "test"), config["training"]["eval_batch_size"], candidates=candidates):
        sequence, proposed = sequence.to(device), proposed.to(device)
        context = process.context(sequence, proposed)
        source = sequence[:, -1]
        for level in range(2):
            projection = process.projection(source, level, context)
            mass = projection.gather(1, proposed).sum(-1)
            inside = process.blocks[level][proposed] == process.blocks[level][source, None]
            entropy = -(projection * projection.clamp_min(1e-30).log()).sum(-1)
            totals[level] += np.array([float(inside.float().mean(-1).sum()), float(mass.sum()), float(entropy.sum())])
        examples += len(sequence)
    write_json(target, {"split": "test", "examples": examples,
               "levels": [{"level": name, "candidate_fraction_in_last_item_block": float(row[0] / examples),
                           "candidate_mass_in_block_projection": float(row[1] / examples),
                           "block_projection_entropy": float(row[2] / examples)}
                          for name, row in zip(("fine", "coarse"), totals)]})


def aggregate_rows(rows):
    groups = {}
    for row in rows:
        suffix = Path(row["report"]).name
        groups.setdefault((row["dataset"], row["experiment"], suffix), []).append(row)
    aggregates = []
    for (dataset, arm, report), group in sorted(groups.items()):
        item = {"dataset": dataset, "experiment": arm, "report_kind": report, "runs": len(group)}
        for key in ("hr@5", "ndcg@5", "hr@10", "ndcg@10"):
            array = np.array([r[key] for r in group])
            item[key] = {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.}
        aggregates.append(item)
    return aggregates


def summary(root):
    rows = []
    for report in sorted((root / "experiments").glob("**/report*.json")):
        cfg = json.loads((report.parent / "resolved_config.json").read_text(encoding="utf-8"))
        result = json.loads(report.read_text(encoding="utf-8"))
        rows.append({"dataset": cfg["dataset"], "experiment": cfg["experiment"], "seed": cfg["seed"],
                     "partition": cfg.get("partition_seed", "default"), "report": str(report.relative_to(root)),
                     **{key: result["test"][key] for key in ("hr@5", "ndcg@5", "hr@10", "ndcg@10")}})
    destination = root / "summary"
    destination.mkdir(exist_ok=True)
    if not rows:
        return
    with (destination / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = aggregate_rows(rows)
    write_json(destination / "aggregate.json", aggregates)
    from benchmark import partition_statistics
    reml = partition_statistics(root)
    write_json(destination / "partition_reml.json", reml)
    contract = json.loads((root / "workflow_config.json").read_text(encoding="utf-8"))
    control_seeds = contract["suite"]["control_seeds"]
    table_groups = {"table1_main": set(MAIN),
                    "table2_factorial": set(FACTORIAL),
                    "table3_auxiliary": {"choicegrow", "prefergrow", "guided_negatives", "popularity"},
                    "table3_forward_toys": {"proposal_root", "proposal_hierarchy", "dynamic_mass", "history_swap", "choicegrow", "uniform_rate", "real_scope"}}
    for name, arms in table_groups.items():
        selected = [r for r in rows if r["experiment"] in arms and
                    (name != "table3_forward_toys" or r["dataset"] == "Toys") and
                    (name != "table3_auxiliary" or r["seed"] in control_seeds) and
                    (Path(r["report"]).name == "report.json" or (name == "table3_auxiliary" and Path(r["report"]).name == "report_score_fusion.json"))]
        write_json(destination / (name + ".json"), aggregate_rows(selected))
    initialization = [r for r in rows if r["experiment"] in {"shared_three", "choicegrow"}
                      and Path(r["report"]).name != "report_score_fusion.json"]
    write_json(destination / "appendix_table4_initialization.json", aggregate_rows(initialization))
    write_json(destination / "appendix_table5_scope.json", {"user_root": [r for r in aggregates if r["experiment"] == "user_root"],
                                                           "hierarchies": reml})
    hazards, scales, scopes = [], [], []
    for folder in sorted((root / "experiments").glob("*/*/*")):
        if not (folder / "resolved_config.json").exists():
            continue
        cfg = json.loads((folder / "resolved_config.json").read_text(encoding="utf-8"))
        identity = {"dataset": cfg["dataset"], "experiment": cfg["experiment"], "seed": cfg["seed"],
                    "partition_seed": cfg.get("partition_seed")}
        if (folder / "hazard_audit.json").exists():
            hazards.append({**identity, **json.loads((folder / "hazard_audit.json").read_text())})
        if "hazard_calibration" in cfg:
            scales.append({**identity, **cfg["hazard_calibration"]})
        if (folder / "scope_diagnostics.json").exists():
            scopes.append({**identity, **json.loads((folder / "scope_diagnostics.json").read_text())})
    write_json(destination / "appendix_table6_corruption.json", hazards)
    write_json(destination / "appendix_table7_matched_exit.json", {"scales": scales,
               "metrics": [r for r in aggregates if r["experiment"].startswith("matched_")]})
    write_json(destination / "appendix_table8_variance.json", reml)
    write_json(destination / "scope_diagnostics.json", scopes)
    for filename, name in (("operators.json", "appendix_table9_operators.json"), ("end_to_end.json", "appendix_table11_cost.json")):
        if (root / "efficiency" / filename).exists():
            shutil.copyfile(root / "efficiency" / filename, destination / name)
    text = ["| Dataset | Experiment | Report | Runs | HR@5 | NDCG@5 | HR@10 | NDCG@10 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in aggregates:
        metrics = [f"{row[k]['mean']:.6f} ± {row[k]['std']:.6f}" for k in ("hr@5", "ndcg@5", "hr@10", "ndcg@10")]
        text.append("| " + " | ".join([row["dataset"], row["experiment"], row["report_kind"], str(row["runs"]), *metrics]) + " |")
    (destination / "tables.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def jobs(suite, datasets, seeds, groups):
    result = {}
    def add(dataset, seed, arm, group, partition=None):
        key = (dataset, seed, arm, partition)
        result.setdefault(key, {"dataset": dataset, "seed": seed, "arm": arm, "groups": [], "partition": partition})["groups"].append(group)
    for dataset in datasets:
        for seed in seeds:
            if "main" in groups:
                for arm in MAIN:
                    add(dataset, seed, arm, "main")
            if "factorial" in groups or "initialization" in groups or "matched_exit" in groups:
                arms = FACTORIAL if "factorial" in groups or "matched_exit" in groups else ("shared_three", "choicegrow")
                for arm in arms:
                    add(dataset, seed, arm, "factorial")
            if "auxiliary" in groups and seed in suite["control_seeds"]:
                for arm in ("prefergrow", "guided_negatives", "popularity"):
                    add(dataset, seed, arm, "auxiliary")
            if "forward" in groups and dataset == "Toys":
                for arm in ("proposal_root", "proposal_hierarchy", "dynamic_mass", "history_swap", "choicegrow", "uniform_rate", "real_scope"):
                    add(dataset, seed, arm, "forward")
            if "matched_exit" in groups:
                for arm in FACTORIAL:
                    add(dataset, seed, "matched_" + arm, "matched_exit")
            if "partitions" in groups and seed in suite["control_seeds"]:
                for partition in suite["partition_seeds"]:
                    for arm in ("shuffle_partition", "real_partition"):
                        add(dataset, seed, arm, "partitions", partition)
    return list(result.values())


def experiment_stage(root, base, suite, datasets, seeds, groups, device, smoke):
    if not set(seeds).issubset(suite["seeds"]):
        raise ValueError(f"--seeds must be chosen from {suite['seeds']}")
    plan = jobs(suite, datasets, seeds, groups)
    write_json(root / "experiment_plan.json", plan)
    chosen, baseline_choices = {}, {}
    for dataset in datasets:
        for seed in seeds:
            if not (source_dir(root, dataset, seed) / "controls.json").exists():
                raise FileNotFoundError(f"Missing candidates for {dataset} seed {seed}; run --stage candidates first")
        if any(j["dataset"] == dataset and j["arm"] not in BASELINES for j in plan):
            chosen[dataset] = tune_dataset(root, base, suite, dataset, device, smoke, seeds[0])
        else:
            chosen[dataset] = config_for(base, dataset, seeds[0], smoke)
    for index, job in enumerate(plan, 1):
        dataset, seed, arm, partition = (job[k] for k in ("dataset", "seed", "arm", "partition"))
        source = source_dir(root, dataset, seed)
        underlying = arm.removeprefix("matched_")
        underlying = "choicegrow" if underlying in {"shuffle_partition", "real_partition", "real_scope"} else underlying
        if arm in BASELINES and arm != "sasrec":
            key = (dataset, arm)
            if key not in baseline_choices:
                baseline_choices[key] = tune_baseline(root, base, suite, dataset, arm,
                         source_dir(root, dataset, seeds[0]), device, smoke, seeds[0])
            config = deepcopy(baseline_choices[key])
            config.update(seed=seed)
            config = attach_inputs(config, source)
        else:
            config = experiment_config(chosen[dataset], suite, dataset, seed, underlying, source, smoke)
        config["experiment"] = arm
        config["full_level_rates"] = chosen[dataset]["diffusion"]["level_rates"]
        config["suite_neighbor_mix"] = chosen[dataset]["destination"]["neighbor_mix"]
        if arm.startswith("real") or partition is not None:
            kind = "real" if arm.startswith("real") else "shuffle"
            partition_seed = partition if partition is not None else base["diffusion"]["partition_seed"]
            structure = root / "scopes" / dataset / f"{kind}_{partition_seed}.pt"
            if not structure.exists():
                print(f"[scope] {dataset} {kind} seed={partition_seed}", flush=True)
                scope_artifact(source, structure, {**config, "suite": suite}, kind, partition_seed)
            config["structure_file"] = str(structure)
            config["partition_seed"] = partition_seed
        if arm.startswith("matched_"):
            calibration_source = source_dir(root, dataset, seeds[0])
            calibration_seed = seeds[0]
            reference = experiment_config(chosen[dataset], suite, dataset, calibration_seed, "shared_root", calibration_source, smoke)
            calibration = experiment_config(chosen[dataset], suite, dataset, calibration_seed, underlying, calibration_source, smoke)
            scale = mean_hazard(calibration_source, reference, device)["mean"] / mean_hazard(calibration_source, calibration, device)["mean"]
            config["diffusion"]["rate_scale"] = scale
            config["hazard_calibration"] = {"split": "val", "source_seed": calibration_seed,
                                           "scope": "dataset_arm", "scale": scale}
        suffix = f"seed_{seed}" + (f"_partition_{partition}" if partition is not None else "")
        directory = root / "experiments" / dataset / arm / suffix
        print(f"[experiment {index}/{len(plan)}] {dataset} {arm} seed={seed}", flush=True)
        if arm == "guided_negatives":
            rows = []
            for weight in suite["guided_negative_weights"]:
                trial = deepcopy(config)
                trial["diffusion"]["guided_negative_weight"] = weight
                result = train_run(root / "auxiliary_selection" / dataset / suffix / str(weight), source, trial, device, test=False)
                rows.append((result["best_validation"]["ndcg@10"], weight))
            config["diffusion"]["guided_negative_weight"] = max(rows)[1]
        train_run(directory, source, config, device)
        if "initialization" in groups and arm in {"shared_three", "choicegrow"}:
            initialization_control(directory, config, device)
        if "auxiliary" in groups and arm == "prefergrow" and seed in suite["control_seeds"]:
            fusion_control(directory, source, config, suite, device)
        if "matched_exit" in groups and arm in FACTORIAL:
            audit = directory / "hazard_audit.json"
            if not audit.exists():
                write_json(audit, mean_hazard(source, config, device, "test", diagnostics=True))
        if ("partitions" in groups or "forward" in groups) and arm in {"choicegrow", "real_scope", "shuffle_partition", "real_partition"}:
            scope_diagnostics(directory, config, device)
        summary(root)
    if "efficiency" in groups:
        from benchmark import run_benchmarks
        run_benchmarks(root, suite, device, smoke)
    summary(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "candidates", "experiments", "all", "summary", "plan"), default="experiments")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--suite", default="paper_suite.json")
    parser.add_argument("--output", default="runs/paper")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args(argv)
    base = json.loads(path_at_root(args.config).read_text(encoding="utf-8"))
    suite = json.loads(path_at_root(args.suite).read_text(encoding="utf-8"))
    datasets = canonical(args.datasets or suite["datasets"])
    seeds = args.seeds or (suite["seeds"][:1] if args.smoke else suite["seeds"])
    groups = args.groups or suite["groups"]
    if set(groups) - set(suite["groups"]):
        raise ValueError("Unknown experiment group")
    if args.smoke:
        suite["partition_seeds"] = suite["partition_seeds"][:1]
        suite["infomap_restarts"] = 1
    if args.skip_tuning:
        suite["tuning"]["enabled"] = False
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(suite["seeds"]):
        raise ValueError(f"Choose distinct --seeds from {suite['seeds']}")
    for section in ("sasrec", "training"):
        if base[section]["epochs"] < 0 or base[section]["patience"] < 1:
            raise ValueError("Epoch limits must be nonnegative and early-stopping patience must be positive")
    if base["training"]["validation_interval"] < 1:
        raise ValueError("Validation interval must be positive")
    root = path_at_root(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "plan":
        print(json.dumps(jobs(suite, datasets, seeds, groups), ensure_ascii=False, indent=2))
        return
    contract = {"base": base, "suite": suite, "smoke": args.smoke}
    contract_path = root / "workflow_config.json"
    if args.stage != "summary":
        if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Workflow settings changed; choose a new --output")
        write_json(contract_path, contract)
    torch.set_num_threads(base["threads"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if args.stage in {"prepare", "all"}:
        preprocessing(root, base, suite, datasets, args.smoke)
    if args.stage in {"candidates", "all"}:
        candidate_stage(root, base, suite, datasets, seeds, device, args.smoke)
    if args.stage in {"experiments", "all"}:
        experiment_stage(root, base, suite, datasets, seeds, groups, device, args.smoke)
    if args.stage == "summary":
        summary(root)


if __name__ == "__main__":
    main()
