"""Exercise complete trainer loops with plateauing validation and no epoch cap."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from train import better, train_diffusion, train_proposer
from experiments import train_run


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


class TinyBaseline(TinyDenoiser):
    def loss(self, sequence, lengths, target):
        return self.weight.square().mean()


def metrics(value):
    return {"ndcg@10": value, "hr@10": value}


class EarlyStoppingTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))
        self.config.update(dataset="Toys", smoke=False)
        self.values = {"seq": torch.tensor([[0, 1], [1, 2]]),
                       "length": torch.tensor([2, 2]), "target": torch.tensor([2, 3])}
        self.device = torch.device("cpu")

    def test_only_validation_ndcg_selects_checkpoint(self):
        self.assertFalse(better({"ndcg@10": .2, "hr@10": .9}, {"ndcg@10": .2, "hr@10": .1}))
        self.assertTrue(better({"ndcg@10": .21, "hr@10": .1}, {"ndcg@10": .2, "hr@10": .9}))

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
             patch("train.metadata", return_value={"item_count": 6}), \
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
            latest = torch.load(directory / "choice_newst.pt", weights_only=True)
            self.assertEqual(latest["epoch"], 8)
            self.assertEqual(sorted(path.name for path in directory.glob("choice_*.pt")),
                             ["choice_best.pt", "choice_newst.pt"])
            self.assertFalse(torch.equal(best["model"]["weight"], latest["model"]["weight"]))

    def test_baseline_tests_best_in_memory_and_recovers_a_test_interruption_without_weights(self):
        self.config.update(method="caser")
        self.config["training"].update(epochs=3, patience=4, validation_interval=1, batch_size=2)
        selected = []
        def evaluate(current, values, config, device, split, streams):
            if split == "val":
                selected.append(current.weight.detach().clone())
                return metrics([.3, .2, .1][(len(selected) - 1) % 3])
            torch.testing.assert_close(current.weight, selected[-3])
            return metrics(.25)
        with tempfile.TemporaryDirectory() as temp, \
             patch("baselines.make_baseline", side_effect=lambda *_: TinyBaseline(6, self.config)), \
             patch("baselines.load_split", return_value=self.values), \
             patch("baselines.evaluate_baseline", side_effect=evaluate), \
             patch("baselines.append_log"):
            directory = Path(temp)
            # A previous process finished training but died before testing its in-memory best.
            (directory / "training.json").write_text('{"best_epoch": 1}')
            train_run(directory, directory / "source", self.config, self.device)
            report = json.loads((directory / "report.json").read_text())
            self.assertEqual(report["best_epoch"], 1)
            self.assertEqual(report["test"]["ndcg@10"], .25)
            self.assertEqual(list(directory.glob("*.pt")), [])
            self.assertEqual(len(selected), 3)
            train_run(directory, directory / "source", self.config, self.device)
            self.assertEqual(len(selected), 3)


if __name__ == "__main__":
    unittest.main()
