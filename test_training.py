"""Exercise complete trainer loops with plateauing validation and no epoch cap."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from train import train_diffusion, train_proposer


class TinyProposer(nn.Module):
    def __init__(self, item_count, config):
        super().__init__()
        self.items = nn.Embedding(item_count, 4)

    def encode(self, sequence, lengths):
        return self.items(sequence[:, -1])


class TinyDenoiser(nn.Module):
    def __init__(self, item_count, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class TinyProcess:
    item_count = 6

    def loss(self, model, sequence, target, proposed):
        return model.weight.square().mean()


def metrics(value):
    return {"ndcg@10": value, "hr@10": value}


class EarlyStoppingTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))
        self.config.update(dataset="Toys", smoke=False)
        self.values = {"seq": torch.tensor([[0, 1], [1, 2]]),
                       "length": torch.tensor([2, 2]), "target": torch.tensor([2, 3])}
        self.device = torch.device("cpu")

    def test_uncapped_proposer_stops_after_validation_plateau_and_keeps_best(self):
        self.config["sasrec"].update(epochs=0, patience=2, batch_size=2, negatives=2)
        with tempfile.TemporaryDirectory() as directory, \
             patch("train.metadata", return_value={"item_count": 6}), \
             patch("train.load_split", return_value=self.values), \
             patch("train.SASRec", TinyProposer), \
             patch("train.evaluate_proposer", side_effect=[metrics(v) for v in [.1, .2, .2, .2]]) as evaluate, \
             patch("train.append_log") as log:
            train_proposer(Path(directory), self.config, self.device)
            best = torch.load(Path(directory) / "sasrec_best.pt", weights_only=True)
            self.assertEqual(evaluate.call_count, 4)
            self.assertEqual(log.call_args.args[1]["epoch"], 4)
            self.assertEqual(best["epoch"], 2)
            self.assertEqual(best["validation"], metrics(.2))

    def test_uncapped_diffusion_uses_epoch_patience_and_retains_best_checkpoint(self):
        self.config["training"].update(epochs=0, patience=4, validation_interval=2, batch_size=2)
        with tempfile.TemporaryDirectory() as directory, \
             patch("train.load_process", return_value=TinyProcess()), \
             patch("train.load_split", return_value=self.values), \
             patch("train.Denoiser", TinyDenoiser), \
             patch("train.evaluate_diffusion", side_effect=[metrics(v) for v in [.1, .2, .2, .2]]) as evaluate, \
             patch("train.append_log"):
            directory = Path(directory)
            torch.save({"train": torch.zeros(2, 1, dtype=torch.long),
                        "val": torch.zeros(2, 1, dtype=torch.long)}, directory / "candidates.pt")
            train_diffusion(directory, self.config, self.device)
            report = json.loads((directory / "training.json").read_text(encoding="utf-8"))
            best = torch.load(directory / "choice_best.pt", weights_only=True)
            self.assertEqual(evaluate.call_count, 4)
            self.assertEqual(report["epochs_completed"], 8)
            self.assertEqual(report["best_epoch"], 4)
            self.assertIsNone(report["epoch_limit"])
            self.assertEqual(report["stop_reason"], "early_stopping")
            self.assertEqual(best["epoch"], 4)


if __name__ == "__main__":
    unittest.main()
