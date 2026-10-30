"""Dependency-aware process queue for independent experiments on one GPU.

Note: .agents/notes/implemented/architecture/2026-10-30-paper-suite.md
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

from data import write_json


def stamp():
    return datetime.now(timezone.utc).isoformat()


def task(name, kind, dataset=None, seed=None, method=None, dependencies=(), **extra):
    return {"id": name, "kind": kind, "dataset": dataset, "seed": seed, "method": method,
            "dependencies": list(dependencies), **extra}


def build_graph(suite, datasets, seeds, groups, stage, smoke=False):
    from experiments import BASELINES, jobs
    tasks = []
    if stage in {"all", "candidates"}:
        for dataset in datasets:
            prep = f"prepare_{dataset}"
            tasks.append(task(prep, "prepare", dataset))
            tasks.append(task(f"semantic_{dataset}", "semantic", dataset, dependencies=[prep]))
            for seed in seeds:
                tasks.append(task(f"candidates_{dataset}_{seed}", "candidates", dataset, seed, dependencies=[prep]))
    if stage in {"all", "experiments"}:
        trial_tasks = []
        for dataset in datasets:
            dependency = [f"candidates_{dataset}_{seeds[0]}"] if stage == "all" else []
            trials = (1 if smoke else suite["tuning"]["trials"][dataset]) if suite["tuning"]["enabled"] and not suite.get("dataset_overrides", {}).get(dataset) else 0
            trial_names = []
            for index in range(trials):
                name = f"search_{dataset}_choice_{index:03d}"
                trial_names.append(name)
                trial_tasks.append(task(name, "choice_trial", dataset, dependencies=dependency, trial=index))
            tasks.append(task(f"tune_{dataset}", "tune", dataset, dependencies=dependency + trial_names))
            for method in BASELINES:
                if method != "sasrec":
                    enabled = suite["baselines"]["tuning"] and suite["tuning"]["enabled"]
                    count = (len(suite["baselines"]["learning_rates"]) * (len(suite["baselines"]["ddsr_skips"]) if method == "ddsr" else 1)) if enabled and not smoke else int(enabled)
                    names = []
                    for index in range(count):
                        name = f"search_{dataset}_{method}_{index:03d}"
                        names.append(name)
                        trial_tasks.append(task(name, "baseline_trial", dataset, method=method, dependencies=dependency, trial=index))
                    tasks.append(task(f"tune_{dataset}_{method}", "baseline_tune", dataset,
                                      method=method, dependencies=dependency + names))
        if stage == "all":
            for item in tasks + trial_tasks:
                if item["kind"] in {"baseline_tune", "baseline_trial"} and item["method"] in {"ddsr", "preferdiff"}:
                    item["dependencies"].append(f"semantic_{item['dataset']}")
        # Interleave datasets rather than filling all slots with Steam's long fits.
        trial_tasks.sort(key=lambda row: (row["trial"], row["method"] or "choicegrow", datasets.index(row["dataset"])))
        tasks.extend(trial_tasks)
        for job in jobs(suite, datasets, seeds, groups):
            dataset, seed, arm = job["dataset"], job["seed"], job["arm"]
            dependency = [] if arm in BASELINES else [f"tune_{dataset}"]
            if stage == "all":
                dependency.append(f"candidates_{dataset}_{seed}")
            if arm in BASELINES and arm != "sasrec":
                dependency.append(f"tune_{dataset}_{arm}")
            partition = job["partition"]
            suffix = f"_partition_{partition}" if partition is not None else ""
            tasks.append(task(f"fit_{dataset}_{arm}_{seed}{suffix}", "experiment", dataset, seed,
                              method=arm, dependencies=dependency, job=job))
    return tasks


def configure_runtime(settings):
    import torch
    torch.set_num_threads(settings["threads_per_worker"])
    torch.set_num_interop_threads(1)
    if settings["tf32"]:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if settings["device"] == "auto" else torch.device(settings["device"])


def tee_output(process, stream, identifier):
    for line in process.stdout:
        stream.write(line)
        stream.flush()
        print(f"[{identifier}] {line}", end="", flush=True)
    process.stdout.close()


def execute_task(root, spec, item):
    from experiments import (ROOT, candidate_stage, experiment_stage, preprocessing,
                             source_dir, tune_baseline, tune_dataset)
    device = configure_runtime(spec)
    base, suite, seeds, smoke = spec["base"], spec["suite"], spec["seeds"], spec["smoke"]
    dataset, kind = item["dataset"], item["kind"]
    print(f"[worker start] {stamp()} {item['id']} device={device}", flush=True)
    if kind == "prepare":
        preprocessing(root, base, suite, [dataset], smoke, semantics=False)
    elif kind == "semantic":
        from semantic import semantic_preprocessing
        from experiments import config_for
        semantic_preprocessing(ROOT / "dataset", root / "data" / dataset,
                               config_for(base, dataset, suite["seeds"][0], smoke), suite["semantic"])
    elif kind == "candidates":
        candidate_stage(root, base, suite, [dataset], [item["seed"]], device, smoke, prepare_data=False)
    elif kind == "tune":
        tune_dataset(root, base, suite, dataset, device, smoke, seeds[0])
    elif kind == "baseline_tune":
        tune_baseline(root, base, suite, dataset, item["method"], source_dir(root, dataset, seeds[0]), device, smoke, seeds[0])
    elif kind in {"choice_trial", "baseline_trial"}:
        from experiments import baseline_search_configs, choice_search_configs, train_run
        source = source_dir(root, dataset, seeds[0])
        index = item["trial"]
        if kind == "choice_trial":
            config = choice_search_configs(base, suite, dataset, seeds[0], source, smoke)[index]
            directory = root / "tuning" / dataset / f"trial_{index:03d}"
        else:
            config = baseline_search_configs(root, base, suite, dataset, item["method"], source, smoke, seeds[0])[index]
            directory = root / "baseline_tuning" / dataset / item["method"] / f"trial_{index:03d}"
        train_run(directory, source, config, device, test=False)
    elif kind == "experiment":
        experiment_stage(root, base, suite, [dataset], seeds, spec["groups"], device, smoke,
                         job_subset=[item["job"]], finalize=False)
    else:
        raise ValueError(kind)
    print(f"[worker complete] {stamp()} {item['id']}", flush=True)


def locked_scope_build(source, destination, config, kind, seed):
    """Serialize construction of a shared scope artifact across seed workers."""
    import fcntl
    from experiment_artifacts import scope_artifact
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / (destination.name + ".lock")).open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not destination.exists():
            scope_artifact(source, destination, config, kind, seed)


def parallel_workflow(root, base, suite, datasets, seeds, groups, device, smoke, workers, stage, tf32):
    """Only the coordinator launches jobs or writes global summaries."""
    import fcntl
    from experiments import jobs, summary
    directory = root / "parallel"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "coordinator.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"A coordinator is already running for {root}") from error
        contract = {"base": base, "suite": suite, "smoke": smoke}
        target = root / "workflow_config.json"
        if target.exists() and json.loads(target.read_text()) != contract:
            raise ValueError("Workflow settings changed; choose a new --output")
        write_json(target, contract)
        spec = {**contract, "datasets": datasets, "seeds": seeds, "groups": groups,
                "device": device, "tf32": tf32, "workers": workers,
                "threads_per_worker": max(1, min(2, (os.cpu_count() or 4) // workers))}
        spec_path = directory / "spec.json"
        write_json(spec_path, spec)
        graph = build_graph(suite, datasets, seeds, groups, stage, smoke)
        write_json(directory / "graph.json", graph)
        write_json(root / "experiment_plan.json", jobs(suite, datasets, seeds, groups))
        completed, failed, running, attempts = set(), {}, {}, {}
        capacity = workers
        for item in graph:
            path = directory / (item["id"] + ".json")
            item["task_file"] = str(path)
            write_json(path, item)
            if (directory / (item["id"] + ".done.json")).exists():
                completed.add(item["id"])
        pending = [item for item in graph if item["id"] not in completed]
        environment = dict(os.environ, OMP_NUM_THREADS=str(spec["threads_per_worker"]),
                           MKL_NUM_THREADS=str(spec["threads_per_worker"]), OPENBLAS_NUM_THREADS="1",
                           PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        print(f"[parallel] tasks={len(graph)} workers={workers} completed={len(completed)}", flush=True)
        while pending or running:
            for identifier, (process, stream, item, reader) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                reader.join()
                stream.close()
                del running[identifier]
                if code == 0:
                    completed.add(identifier)
                    write_json(directory / (identifier + ".done.json"), {"completed_at": stamp(), "exit_code": 0})
                    print(f"[complete {len(completed)}/{len(graph)}] {identifier}", flush=True)
                    if item["kind"] == "experiment":
                        summary(root)
                else:
                    log = (directory / (identifier + ".log")).read_text(errors="replace")
                    if "out of memory" in log.lower() and attempts[identifier] < 3 and capacity > 1:
                        capacity -= 1
                        pending.append(item)
                        print(f"[retry OOM] {identifier}; concurrency={capacity}", flush=True)
                    else:
                        failed[identifier] = code
                        print(f"[failed exit={code}] {identifier}; see parallel/{identifier}.log", flush=True)
            priority = {"prepare": 0, "semantic": 1, "candidates": 2, "tune": 3, "baseline_tune": 3,
                        "experiment": 4, "choice_trial": 5, "baseline_trial": 5}
            ordered = sorted(pending, key=lambda row: (priority[row["kind"]],
                             seeds.index(row["seed"]) if row["seed"] in seeds else row.get("trial", 0),
                             datasets.index(row["dataset"])))
            for item in ordered:
                limit_file = directory / "worker_limit.json"
                requested = json.loads(limit_file.read_text()).get("workers", workers) if limit_file.exists() else workers
                limit = max(1, min(capacity, int(requested)))
                if len(running) >= limit:
                    break
                if not set(item["dependencies"]).issubset(completed):
                    continue
                if item["kind"] == "semantic" and any(value[2]["kind"] == "semantic" for value in running.values()):
                    continue
                identifier = item["id"]
                stream = (directory / (identifier + ".log")).open("a", buffering=1, encoding="utf-8")
                process = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                          "--worker", item["task_file"], "--spec", str(spec_path), "--output", str(root)],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment,
                          text=True, encoding="utf-8", errors="replace")
                reader = threading.Thread(target=tee_output, args=(process, stream, identifier), daemon=True)
                reader.start()
                running[identifier] = (process, stream, item, reader)
                attempts[identifier] = attempts.get(identifier, 0) + 1
                pending.remove(item)
                print(f"[launch pid={process.pid}] {identifier}", flush=True)
            write_json(directory / "status.json", {"updated_at": stamp(), "state": "running",
                       "total": len(graph), "completed": len(completed), "failed": failed,
                       "running": {key: value[0].pid for key, value in running.items()},
                       "pending": len(pending)})
            if not running and pending:
                write_json(directory / "status.json", {"updated_at": stamp(), "state": "failed",
                           "total": len(graph), "completed": len(completed), "failed": failed,
                           "running": {}, "pending": len(pending)})
                raise RuntimeError(f"Dependency queue blocked by failures: {failed}")
            if running:
                time.sleep(2)
        if failed:
            write_json(directory / "status.json", {"updated_at": stamp(), "state": "failed",
                       "total": len(graph), "completed": len(completed), "failed": failed,
                       "running": {}, "pending": 0})
            raise RuntimeError(f"Training tasks failed: {failed}")
        if stage in {"all", "experiments"} and "efficiency" in groups:
            # No concurrent fitting, semantic construction, or CUDA worker here.
            from benchmark import run_benchmarks
            benchmark_device = configure_runtime(spec)
            print("[efficiency] exclusive GPU after all training tasks", flush=True)
            run_benchmarks(root, suite, benchmark_device, smoke)
            summary(root)
        write_json(directory / "status.json", {"updated_at": stamp(), "state": "complete",
                   "total": len(graph), "completed": len(completed), "failed": {}, "running": {}, "pending": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        execute_task(Path(args.output), json.loads(Path(args.spec).read_text()), json.loads(Path(args.worker).read_text()))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
