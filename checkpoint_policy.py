"""Retain ChoiceGrow best/newst; release temporary baseline weights after their consumers."""
import json
from pathlib import Path


MODEL_FILES = ("baseline_best.pt", "choice_best.pt", "choice_newst.pt", "sasrec_best.pt")


def prune_run(directory, config, keep_for_efficiency=False):
    directory = Path(directory)
    keep = {"choice_best.pt", "choice_newst.pt"} if config.get("method", "choicegrow") == "choicegrow" else set()
    if keep_for_efficiency:
        keep.add("choice_best.pt")
    removed = []
    for name in MODEL_FILES:
        for path in directory.rglob(name):
            if path.parent == directory and name in keep:
                continue
            path.unlink(missing_ok=True)
            removed.append(str(path))
    return removed


def efficiency_pending(root, suite, groups, dataset, arm, seed):
    benchmark = suite["benchmark"]
    return ("efficiency" in groups and arm in {"choicegrow", "prefergrow"}
            and dataset == benchmark["end_to_end_dataset"] and seed == benchmark["end_to_end_seed"]
            and not (Path(root) / "efficiency" / f"end_to_end_{dataset}_{arm}_{seed}.json").exists())


def cleanup_completed(root, suite, groups, finished=False):
    """Also clean completed legacy workers, while leaving active fits and appendix inputs intact."""
    root = Path(root)
    plan_file = root / "experiment_plan.json"
    if not plan_file.exists():
        return []
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    removed, all_complete = [], True
    for job in plan:
        dataset, arm, seed, partition = (job[key] for key in ("dataset", "arm", "seed", "partition"))
        suffix = f"_partition_{partition}" if partition is not None else ""
        directory = root / "experiments" / dataset / arm / f"seed_{seed}{suffix}"
        done = root / "parallel" / f"fit_{dataset}_{arm}_{seed}{suffix}.done.json"
        complete = (directory / "evaluation_complete.json").exists() or done.exists()
        if not complete:
            all_complete = False
            continue
        config_file = directory / "resolved_config.json"
        if config_file.exists():
            config = json.loads(config_file.read_text(encoding="utf-8"))
            removed += prune_run(directory, config, efficiency_pending(root, suite, groups, dataset, arm, seed))
    # Candidate-only stage must keep SASRec for later fusion and online timing.
    if plan and all_complete and (finished or "efficiency" not in groups):
        for path in (root / "candidates").rglob("sasrec_best.pt"):
            path.unlink(missing_ok=True)
            removed.append(str(path))
    return removed
