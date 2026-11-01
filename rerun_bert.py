"""Rerun only BERT4Rec using existing prepared splits and the author protocol."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from data import ALIASES, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Previous experiment root containing data/<dataset>/prepared")
    parser.add_argument("--output", required=True, help="New output directory; old results stay intact")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--suite", default="paper_suite.json")
    parser.add_argument("--datasets", nargs="+", default=["Beauty", "Toys", "Sports"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    source, root = Path(args.source).resolve(), Path(args.output).resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError("Use an independent output directory beside the source run")
    if args.workers < 1 or args.threads_per_worker < 1:
        raise ValueError("Worker and thread counts must be positive")
    from experiments import arm_config, config_for, frozen_config, train_run
    base = json.loads(Path(args.config).read_text(encoding="utf-8"))
    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    datasets = [ALIASES[value.lower()] for value in args.datasets]
    for dataset in datasets:
        if not (source / "data" / dataset / "prepared" / "metadata.json").exists():
            raise FileNotFoundError(f"Missing prepared data for {dataset}")
    if args.worker:
        import torch
        dataset, seed = args.worker[0], int(args.worker[1])
        torch.set_num_threads(args.threads_per_worker)
        torch.set_num_interop_threads(1)
        if args.tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.allow_tf32 = True
        config = arm_config(config_for(base, dataset, seed, args.smoke), "bert4rec", suite)
        config["threads"] = args.threads_per_worker
        directory = root / "experiments" / dataset / "bert4rec" / f"seed_{seed}"
        train_run(directory, source / "data" / dataset, config, torch.device(args.device))
        write_json(directory / "evaluation_complete.json", {"complete": True})
        return
    plan = [(dataset, seed) for seed in args.seeds for dataset in datasets]
    frozen_config(root, {"source": str(source), "base": base, "suite": suite,
                        "datasets": datasets, "seeds": args.seeds, "smoke": args.smoke,
                        "threads": args.threads_per_worker, "tf32": args.tf32, "device": args.device})
    write_json(root / "workflow_config.json", {"base": base, "suite": suite, "smoke": args.smoke})
    logs = root / "parallel"
    logs.mkdir(exist_ok=True)
    pending, completed = [], []
    for dataset, seed in plan:
        name = f"bert_{dataset}_{seed}"
        if (root / "experiments" / dataset / "bert4rec" / f"seed_{seed}" / "report.json").exists():
            completed.append(name)
        else:
            pending.append((dataset, seed))
    running, failed = {}, {}
    from parallel import tee_output
    while pending or running:
        while pending and len(running) < args.workers:
            dataset, seed = pending.pop(0)
            name = f"bert_{dataset}_{seed}"
            command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                       "--source", str(source), "--output", str(root), "--config", args.config,
                       "--suite", args.suite, "--datasets", dataset, "--worker", dataset, str(seed),
                       "--device", args.device, "--threads-per-worker", str(args.threads_per_worker)]
            if args.tf32:
                command.append("--tf32")
            if args.smoke:
                command.append("--smoke")
            env = dict(os.environ, OMP_NUM_THREADS=str(args.threads_per_worker),
                       MKL_NUM_THREADS=str(args.threads_per_worker), OPENBLAS_NUM_THREADS="1")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, env=env)
            stream = (logs / f"{name}.log").open("a", encoding="utf-8")
            thread = threading.Thread(target=tee_output, args=(process, stream, name), daemon=True)
            thread.start()
            running[name] = (process, thread, stream)
        for name, (process, thread, stream) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            thread.join()
            stream.close()
            if code:
                failed[name] = code
            else:
                completed.append(name)
            del running[name]
        write_json(logs / "status.json", {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": "running" if running or pending else "failed" if failed else "complete",
            "total": len(plan), "completed": len(completed), "failed": failed,
            "running": {name: row[0].pid for name, row in running.items()}, "pending": len(pending),
        })
        if running:
            time.sleep(1)
    from experiments import summary
    summary(root)
    if failed:
        raise SystemExit(f"Failed BERT4Rec jobs: {failed}; inspect {logs}")
    print(f"[complete] {len(completed)} BERT4Rec runs -> {root / 'summary' / 'tables.md'}", flush=True)


if __name__ == "__main__":
    main()
