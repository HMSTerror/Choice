"""Baseline objectives, semantic posterior, and leakage boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from baselines import Caser, DDSR, DiffRec, GRU4Rec, BERT4Rec, PreferDiff
from experiments import arm_config, config_for, train_run
from data import write_json


ROOT = Path(__file__).resolve().parent


class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.suite = json.loads((ROOT / "paper_suite.json").read_text())
        self.base = json.loads((ROOT / "config.json").read_text())
        self.base.update(hidden_size=8, history_length=4)
        self.base["sasrec"].update(dropout=0.)
        self.sequence = torch.tensor([[12, 12, 1, 2], [12, 3, 4, 5]])
        self.lengths = torch.tensor([2, 3])
        self.target = torch.tensor([6, 7])
        self.artifact = {"features": torch.randn(12, 8), "codebooks": 2, "codebook_size": 4,
                         "codes": torch.cat((torch.randint(4, (12, 2)), torch.full((1, 2), 4))),
                         "centroids": torch.randn(2, 4, 2)}

    def config(self, method):
        return arm_config(config_for(self.base, "Toys", 100, True), method, self.suite)

    def test_distinct_baselines_have_finite_objectives_gradients_and_full_catalog_scores(self):
        for method, constructor in (("caser", Caser), ("gru4rec", GRU4Rec), ("bert4rec", BERT4Rec),
                                    ("diffrec", DiffRec), ("preferdiff", PreferDiff), ("ddsr", DDSR)):
            with self.subTest(method=method):
                args = (self.artifact,) if method in {"preferdiff", "ddsr"} else ()
                model = constructor(12, self.config(method), *args)
                loss = model.loss(self.sequence, self.lengths, self.target)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
                with torch.no_grad():
                    scores = model.eval()(self.sequence, self.lengths)
                self.assertEqual(scores.shape, (2, 12))
                self.assertTrue(bool(torch.isfinite(scores).all()))

    def test_ddsr_skip_posterior_matches_dense_joint_marginalization(self):
        model = DDSR(12, self.config("ddsr"), self.artifact)
        predicted = torch.rand(2, 4, 2, 4).softmax(-1)
        current = torch.randint(4, (2, 4, 2))
        step, previous = 3, 1
        actual = model.reverse_probabilities(predicted, current, step, previous)
        k = model.k
        a = model.cumulative[previous]
        ratio = model.cumulative[step] / a
        cumulative = a * torch.eye(k) + (1 - a) / k
        transition = ratio * torch.eye(k) + (1 - ratio) / k
        for b in range(2):
            for p in range(4):
                for level in range(2):
                    marginal = predicted[b, p, level] @ cumulative
                    joint = marginal * transition[:, current[b, p, level]]
                    torch.testing.assert_close(actual[b, p, level], joint / joint.sum())

    def test_diffrec_is_history_interaction_vector_and_excludes_padding(self):
        model = DiffRec(12, self.config("diffrec"))
        x = model.interaction_vector(self.sequence)
        self.assertEqual(x.sum().item(), 5)
        torch.testing.assert_close(x, model.interaction_vector(self.sequence.flip(-1)))
        train = model.interaction_vector(self.sequence, self.target)
        self.assertTrue(bool((train.gather(1, self.target[:, None]) == 1).all()))
        self.assertTrue(bool((x.gather(1, self.target[:, None]) == 0).all()))

    def test_preferdiff_semantic_features_are_fixed_during_training(self):
        model = PreferDiff(12, self.config("preferdiff"), self.artifact)
        model.loss(self.sequence, self.lengths, self.target).backward()
        self.assertFalse(model.history.items.weight.requires_grad)
        self.assertIsNone(model.history.items.weight.grad)
        self.assertIsNotNone(model.denoise.weight.grad)

    def test_baseline_search_does_not_test_and_formal_fit_tests_once_after_training(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            config = self.config("gru4rec")
            events = []
            def train(folder, *_):
                events.append("train")
                write_json(folder / "training.json", {"best_validation": {"ndcg@10": .1}})
            def test(folder, *_):
                events.append("test")
                write_json(folder / "report.json", {"test": {"ndcg@10": .1}})
            with patch("experiments.train_baseline", side_effect=train), patch("experiments.test_baseline", side_effect=test):
                train_run(directory / "search", directory, config, torch.device("cpu"), test=False)
                self.assertEqual(events, ["train"])
                train_run(directory / "formal", directory, config, torch.device("cpu"))
                train_run(directory / "formal", directory, config, torch.device("cpu"))
                self.assertEqual(events, ["train", "train", "test"])


if __name__ == "__main__":
    unittest.main()
