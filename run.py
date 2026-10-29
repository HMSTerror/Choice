"""A single entry point for the complete minimal Choice experiment."""
# Note: .agents/notes/implemented/architecture/2026-10-29-minimal-choice.md
import argparse
import json
from pathlib import Path

import torch

from data import ALIASES, prepare, write_json
from train import build_artifacts, final_evaluation, train_diffusion, train_proposer


ROOT = Path(__file__).resolve().parent
STAGES = ("prepare", "proposer", "artifacts", "train", "evaluate")


def local_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Toys",
                        choices=(*ALIASES, "Toys", "Sports", "Beauty", "Steam", "ML1M", "ATG", "ASO"))
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", help="A run directory, relative to this project")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, help="Diffusion epoch limit; 0 means validation early stopping only")
    parser.add_argument("--sasrec-epochs", type=int, help="SASRec epoch limit; 0 means validation early stopping only")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(local_path(args.config).read_text(encoding="utf-8"))
    config["dataset"] = ALIASES[args.dataset.lower()]
    config["smoke"] = args.smoke
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.sasrec_epochs is not None:
        config["sasrec"]["epochs"] = args.sasrec_epochs
    if args.smoke:
        config["sasrec"].update(epochs=1, batch_size=8)
        config["training"].update(epochs=1, batch_size=8, eval_batch_size=8, validation_interval=1,
                                  validation_streams=1, test_streams=1)
    if config["training"]["epochs"] < 0 or config["sasrec"]["epochs"] < 0:
        raise ValueError("Epoch limits must be nonnegative; 0 means early stopping only")
    if config["sasrec"]["patience"] < 1:
        raise ValueError("SASRec patience must be positive")
    if config["training"]["validation_interval"] < 1 or config["training"]["patience"] < 1:
        raise ValueError("Validation interval and patience must be positive")
    torch.set_num_threads(config["threads"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu")
    name = f"{'smoke' if args.smoke else 'experiment'}_{config['dataset'].lower()}_s{config['seed']}"
    output = local_path(args.output or f"runs/{name}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "resolved_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Output directory belongs to different settings; choose a new --output")
    write_json(config_path, config)
    actions = {
        "prepare": lambda: prepare(ROOT / "dataset", output, config["dataset"], config),
        "proposer": lambda: train_proposer(output, config, device),
        "artifacts": lambda: build_artifacts(output, config, device),
        "train": lambda: train_diffusion(output, config, device),
        "evaluate": lambda: final_evaluation(output, config, device),
    }
    for stage in STAGES if args.stage == "all" else (args.stage,):
        print(f"[{stage}] dataset={config['dataset']} device={device} output={output}", flush=True)
        actions[stage]()


if __name__ == "__main__":
    main()
