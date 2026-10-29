"""Numerical and data-contract checks; run with python -m unittest -v test_core."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from data import read_source
from diffusion import ChoiceProcess, Context, random_partition
from models import Denoiser
from neighbors import build_neighbors


ROOT = Path(__file__).resolve().parent


def fixture():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    config["dataset"] = "Toys"
    base = torch.tensor([1., 2., 3., 4., 5., 6.], dtype=torch.float64)
    base /= base.sum()
    artifact = {"base": base, "blocks": torch.tensor([[0, 0, 1, 1, 2, 2],
                 [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0]]),
                 "neighbor_ids": torch.arange(6)[:, None],
                 "neighbor_weights": torch.ones(6, 1, dtype=torch.float64)}
    process = ChoiceProcess(artifact, config)
    rho = torch.tensor([[.05, .4, .1, .2, .15, .1]], dtype=torch.float64)
    context = Context(rho, tuple(process.block_sum(rho, level) for level in range(3)))
    return process, context, config


def dense_generator(process, context):
    rows = Context(context.rho.expand(6, -1), tuple(mass.expand(6, -1) for mass in context.block_mass))
    identity = torch.eye(6, dtype=torch.float64)
    generator = process.reset_rate * (process.base.expand(6, -1) - identity)
    for level, rate in enumerate(process.rates):
        generator += rate * (process.projection(torch.arange(6), level, rows) - identity)
    return generator, rows


class CoreTests(unittest.TestCase):
    def test_exact_forward_and_terminal_match_dense_exponential(self):
        process, context, _ = fixture()
        generator, rows = dense_generator(process, context)
        for sigma in (.001, .2, 10.):
            expected = torch.matrix_exp(generator * sigma)
            actual = process.forward_probability(torch.arange(6), torch.full((6,), sigma, dtype=torch.float64), rows)
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-11)
            torch.testing.assert_close(process.terminal_reference(sigma, context)[0],
                                       process.base @ expected, atol=2e-12, rtol=2e-11)
            torch.testing.assert_close(actual.sum(-1), torch.ones(6, dtype=torch.float64))

    def test_reverse_direction_and_entropy_gradient(self):
        process, context, _ = fixture()
        generator, rows = dense_generator(process, context)
        expected = generator.T.clone()
        expected.fill_diagonal_(0.)
        torch.testing.assert_close(process.incoming_rates(torch.arange(6), rows), expected)
        probability = process.forward_probability(torch.tensor([3]), torch.tensor([.2], dtype=torch.float64), context)
        current = torch.tensor([1])
        reference = (probability.log() - probability[:, 1:2].log()).detach().requires_grad_()
        loss = process.entropy(reference, probability, current, context).sum()
        torch.testing.assert_close(loss, torch.tensor(0., dtype=torch.float64))
        torch.testing.assert_close(torch.autograd.grad(loss, reference)[0], torch.zeros_like(reference))

    def test_destination_keeps_fixed_mass_before_neighbor_mixing(self):
        process, _, config = fixture()
        process.destination = deepcopy(config["destination"])
        sequence, candidates = torch.tensor([[6, 6, 0, 1]]), torch.tensor([[2, 3]])
        process.destination["neighbor_mix"] = 0.
        plain = process.context(sequence, candidates).rho
        self.assertAlmostEqual(float(plain[:, [2, 3]].sum()), config["destination"]["candidate_mass"])
        process.destination["neighbor_mix"] = .02
        mixed = process.context(sequence, candidates).rho
        self.assertTrue(torch.all(mixed > 0))
        torch.testing.assert_close(mixed.sum(-1), torch.ones(1, dtype=torch.float64))
        self.assertGreater(float(mixed[0, 1]), float(plain[0, 1]))

    def test_neighbor_lag_weights_self_prior_and_fallback(self):
        ids, weights = build_neighbors(torch.tensor([[5, 0, 1, 2]]), torch.tensor([3]), 5,
                                       count=3, lags=3, self_prior=1.)
        distribution = torch.zeros(5, 5).scatter_add(1, ids, weights)
        self.assertAlmostEqual(float(distribution[2, 3]), .5)
        self.assertAlmostEqual(float(distribution[1, 3]), 1 / 3, places=6)
        self.assertAlmostEqual(float(distribution[0, 3]), .25)
        self.assertEqual(float(distribution[4, 4]), 1.)

    def test_random_partition_is_nested_and_seeded(self):
        blocks = random_partition(2099, 64, 1024, 100)
        torch.testing.assert_close(blocks, random_partition(2099, 64, 1024, 100))
        for block in torch.unique(blocks[0]):
            self.assertEqual(torch.unique(blocks[1, blocks[0] == block]).numel(), 1)
        sizes = torch.bincount(blocks[0])
        self.assertTrue(torch.all(sizes[:-1] == 64))
        self.assertEqual(int(sizes[-1]), 51)

    def test_text_and_dataframe_preprocessing_have_same_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "train_data.txt").write_text("1 3 5\n2 4\n", encoding="utf-8")
            text, _ = read_source(directory, "train", 5, 4)
            pd.DataFrame({"seq": [[5, 5, 0, 2], [5, 5, 5, 1]], "len_seq": [2, 1],
                          "next": [4, 3]}).to_pickle(directory / "val_data.df")
            frame, _ = read_source(directory, "val", 5, 4)
            for key in text:
                np.testing.assert_array_equal(text[key], frame[key])

    def test_denoiser_has_finite_anchored_scores_and_no_auxiliary_parameters(self):
        _, _, config = fixture()
        model = Denoiser(6, config)
        sequence = torch.tensor([[6, 6, 6, 6, 6, 6, 6, 6, 0, 1]])
        current, sigma = torch.tensor([2]), torch.tensor([.1])
        result = model(sequence, current, sigma)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(float(result[0, 2]), 0.)
        self.assertFalse(any("aux" in name for name, _ in model.named_parameters()))


if __name__ == "__main__":
    unittest.main()
