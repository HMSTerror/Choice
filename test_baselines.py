"""Baseline objectives, semantic posterior, and leakage boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from baselines import (Caser, DDSR, DiffRec, GRU4Rec, BERT4Rec, PreferDiff,
                       make_baseline_optimizer, sampled_binary_cross_entropy,
                       baseline_learning_rate, top1_loss, PaperAdagrad, BertAdam)
from experiments import arm_config, config_for, train_run
from data import write_json
from semantic import whiten_features


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

    def test_caser_uses_three_negatives_and_gru_top1_uses_batch_targets(self):
        model = Caser(12, self.config("caser"))
        with patch("baselines.torch.randint", return_value=torch.zeros(2, 3, dtype=torch.long)) as draw:
            loss = model.loss(self.sequence, self.lengths, self.target)
        self.assertTrue(bool(torch.isfinite(loss)))
        draw.assert_called_once_with(11, (2, 3), device=self.target.device)
        model = GRU4Rec(12, self.config("gru4rec"))
        with patch("baselines.torch.randint") as draw:
            loss = model.loss(self.sequence, self.lengths, self.target)
        draw.assert_not_called()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertLess(top1_loss(torch.tensor([[1., -1.], [-1., 1.]])),
                        top1_loss(torch.tensor([[-1., 1.], [1., -1.]])))

    def test_original_model_optimizers_and_bert_decay(self):
        for method, constructor, optimizer_type in (("gru4rec", GRU4Rec, PaperAdagrad),
                       ("caser", Caser, torch.optim.Adam), ("bert4rec", BERT4Rec, BertAdam)):
            config = self.config(method)
            optimizer = make_baseline_optimizer(constructor(12, config), config)
            self.assertIsInstance(optimizer, optimizer_type)
        settings = self.config("bert4rec")["training"]
        self.assertEqual(baseline_learning_rate(settings, 0), 0.)
        self.assertAlmostEqual(baseline_learning_rate(settings, 50), settings["learning_rate"] / 2)
        self.assertAlmostEqual(baseline_learning_rate(settings, 200000), settings["learning_rate"] / 2)
        self.assertEqual(baseline_learning_rate(settings, 400000), 0.)

    def test_fixed_model_profiles_do_not_inherit_choice_width_or_lr(self):
        for method, width, lr, batch in (("caser", 50, .001, 100), ("gru4rec", 100, .01, 50),
                  ("bert4rec", 64, .0001, 256), ("preferdiff", 3072, .0001, 256),
                  ("diffrec", 8, .0001, 400), ("ddsr", 8, .0001, 2048)):
            config = arm_config(config_for(self.base, "Toys", 100, False), method, self.suite)
            self.assertEqual(config["hidden_size"], width)
            self.assertEqual(config["training"]["learning_rate"], lr)
            self.assertEqual(config["training"]["batch_size"], batch)
        beauty = arm_config(config_for(self.base, "Beauty", 100), "preferdiff", self.suite)
        self.assertEqual(beauty["baseline"]["preferdiff"]["guidance"], 6.)
        self.assertEqual(beauty["baseline"]["preferdiff"]["lambda"], .8)

    def test_paper_optimizer_updates_use_author_epsilon_and_moment_conventions(self):
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        parameter.grad = torch.tensor([2.])
        optimizer = PaperAdagrad([parameter], lr=.1, eps=1.)
        optimizer.step()
        torch.testing.assert_close(parameter, torch.tensor([1 - .2 / 5 ** .5]))
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        parameter.grad = torch.tensor([2.])
        optimizer = BertAdam([{"params": [parameter], "weight_decay": .01}], lr=.1, eps=1e-6)
        optimizer.step()
        expected = 1 - .1 * (.2 / (.004 ** .5 + 1e-6) + .01)
        torch.testing.assert_close(parameter, torch.tensor([expected]))

    def test_caser_reorders_left_padded_history_before_convolution(self):
        model = Caser(12, self.config("caser")).eval()
        captured = []
        handle = model.vertical.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
        model.encode(self.sequence, self.lengths)
        handle.remove()
        expected = model.items(torch.tensor([[1, 2, 12, 12], [3, 4, 5, 12]]))
        valid = torch.arange(4)[None, :] < self.lengths[:, None]
        expected = expected * valid[..., None]
        torch.testing.assert_close(captured[0].squeeze(1), expected)

    def test_sampled_bce_excludes_positive_item_and_matches_pointwise_loss(self):
        state = torch.tensor([[1., 2.]])
        item_embeddings = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        target = torch.tensor([3])
        with patch("baselines.torch.randint", return_value=torch.tensor([[3, 3]])):
            actual = sampled_binary_cross_entropy(state, item_embeddings, target, 2)
        negative_ids = torch.tensor([[4, 4]])
        positive = (state * item_embeddings[target]).sum(-1)
        negative = (state[:, None] * item_embeddings[negative_ids]).sum(-1)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))
        expected += torch.nn.functional.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative))
        torch.testing.assert_close(actual, expected)

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

    def test_native_whitening_preserves_width_and_fits_training_items_only(self):
        torch.manual_seed(41)
        features = torch.randn(12, 6)
        seen = torch.arange(9)
        whitened = whiten_features(features, seen)
        changed = features.clone()
        changed[9:] += 1000
        torch.testing.assert_close(whitened[seen], whiten_features(changed, seen)[seen])
        self.assertEqual(whitened.shape, features.shape)
        torch.testing.assert_close(whitened[seen].mean(0), torch.zeros(6), atol=1e-5, rtol=0)
        torch.testing.assert_close(whitened[seen].T @ whitened[seen] / 8, torch.eye(6), atol=1e-5, rtol=1e-5)

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
