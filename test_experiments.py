"""Protocol, leakage and mathematical regression tests for the paper workflow."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from data import prepared_root, write_json
from diffusion import ChoiceProcess, random_partition
from experiment_artifacts import control_candidates
from experiments import MAIN, arm_config, jobs, train_run
from models import Denoiser
from prefergrow import PreferGrowProcess
from benchmark import partition_statistics


ROOT = Path(__file__).resolve().parent


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config.json").read_text())
        self.config.update(dataset="Toys", smoke=True, hidden_size=8)
        self.config["training"].update(heads=2, dropout=0.)
        self.config["destination"].update(neighbor_mix=0., candidate_mass=.02)
        self.suite = json.loads((ROOT / "paper_suite.json").read_text())
        n = 12
        base = torch.arange(1, n + 1).float()
        self.artifact = {"base": base / base.sum(), "blocks": random_partition(n, 2, 4, 10),
                         "neighbor_ids": torch.arange(n)[:, None], "neighbor_weights": torch.ones(n, 1)}
        self.history = torch.tensor([[12, 12, 1, 2], [12, 3, 4, 5]])
        self.proposed = torch.tensor([[0, 1, 2], [3, 4, 5]])

    def test_full_plan_covers_paper_groups_without_duplicate_training(self):
        plan = jobs(self.suite, self.suite["datasets"], self.suite["seeds"], self.suite["groups"])
        self.assertEqual(len(plan), 468)
        keys = {(r["dataset"], r["arm"], r["seed"], r["partition"]) for r in plan}
        self.assertEqual(len(keys), len(plan))
        main = [r for r in plan if "main" in r["groups"]]
        self.assertEqual({r["arm"] for r in main}, set(MAIN))
        self.assertEqual(len(main), 108)
        self.assertEqual({r["seed"] for r in main}, set(self.suite["main_seeds"]))
        factorial = [r for r in plan if "factorial" in r["groups"]]
        self.assertEqual({r["seed"] for r in factorial}, set(self.suite["seeds"]))
        for row in plan:
            if "forward" in row["groups"]:
                self.assertEqual(row["dataset"], "Toys")

    def test_initialization_cross_uses_three_level_checkpoints(self):
        plan = jobs(self.suite, ["Toys"], [100], ["initialization"])
        self.assertEqual({row["arm"] for row in plan}, {"shared_three", "choicegrow"})

    def test_factored_arms_keep_event_clock_and_change_named_factor(self):
        total = sum(self.config["diffusion"]["level_rates"])
        for arm in ("shared_root", "user_root", "shared_three", "choicegrow", "uniform_rate"):
            config = arm_config(self.config, arm, self.suite)
            self.assertAlmostEqual(sum(config["diffusion"]["level_rates"]), total)
            self.assertEqual(config["diffusion"]["reset_rate"], self.config["diffusion"]["reset_rate"])
            process = ChoiceProcess(self.artifact, config)
            context = process.context(self.history, self.proposed)
            torch.testing.assert_close(context.rho.sum(-1), torch.ones(2))
            if arm.startswith("shared"):
                torch.testing.assert_close(context.rho, process.base.expand(2, -1))
        config = arm_config(self.config, "proposal_root", self.suite)
        process = ChoiceProcess(self.artifact, config)
        context = process.context(self.history, self.proposed)
        torch.testing.assert_close(context.rho.gather(1, self.proposed), torch.full((2, 3), .02 / 3))

    def test_rate_scaling_matches_dense_generator_and_terminal(self):
        self.config["diffusion"]["rate_scale"] = 1.7
        self.artifact["base"] = self.artifact["base"].double()
        self.artifact["base"] /= self.artifact["base"].sum()
        process = ChoiceProcess(self.artifact, self.config).double()
        context = process.context(self.history[:1], self.proposed[:1])
        n = process.item_count
        q = process.reset_rate * process.base.expand(n, -1).clone()
        for level, rate in enumerate(process.rates):
            members = process.blocks[level, :, None] == process.blocks[level, None, :]
            q += rate * context.rho * members / context.block_mass[level].T
        q -= (sum(process.rates) + process.reset_rate) * torch.eye(n, dtype=torch.double)
        dense = torch.matrix_exp(.8 * q)
        source, sigma = torch.tensor([3]), torch.tensor([.8], dtype=torch.double)
        torch.testing.assert_close(process.forward_probability(source, sigma, context), dense[3:4], atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(process.terminal_reference(.8, context), process.base[None] @ dense, atol=1e-9, rtol=1e-8)

    def test_candidate_controls_do_not_read_validation_or_test_targets(self):
        candidates = {s: torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]) for s in ("train", "val", "test")}
        values = {s: {"seq": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]), "target": torch.arange(4)} for s in candidates}
        counts = torch.arange(12)
        first = control_candidates(candidates, values, counts, 10, window=4)
        changed = deepcopy(values)
        changed["val"]["target"] += 1000
        changed["test"]["target"] += 1000
        second = control_candidates(candidates, changed, counts, 10, window=4)
        for a, b in zip(first, second):
            for split in candidates:
                torch.testing.assert_close(a[split], b[split])
                self.assertTrue(all(len(set(row)) == 2 for row in a[split].tolist()))

    def test_dynamic_mass_receives_gradient_and_keeps_full_support(self):
        config = arm_config(self.config, "dynamic_mass", self.suite)
        config["history_length"] = 4
        model = Denoiser(12, config).eval()
        process = ChoiceProcess(self.artifact, config)
        context = process.context(self.history, self.proposed, model)
        self.assertTrue(bool((context.rho > 0).all()))
        loss = process.loss(model, self.history, torch.tensor([6, 7]), self.proposed)
        loss.backward()
        self.assertTrue(torch.isfinite(model.mass_head.bias.grad).all())
        self.assertGreater(float(model.mass_head.bias.grad.abs().sum()), 0.)

    def test_prefergrow_closed_form_matches_dense_exponential_with_extra_state(self):
        config = arm_config(self.config, "prefergrow", self.suite)
        process = PreferGrowProcess(self.artifact, config).double()
        context = process.context(self.history[:1], self.proposed[:1])
        n = process.item_count
        q = context.rho.expand(n, -1) - torch.eye(n, dtype=torch.double)
        sigma = torch.tensor([.7], dtype=torch.double)
        p = process.forward_probability(torch.tensor([2]), sigma, context)
        torch.testing.assert_close(p, torch.matrix_exp(.7 * q)[2:3], atol=1e-9, rtol=1e-8)
        current = torch.tensor([n - 1])
        torch.testing.assert_close(process.incoming_rates(current, context), q[:, n - 1][None].scatter(1, current[:, None], 0.))

    def test_train_run_tests_immediately_and_reuses_complete_results(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            events = []
            def train(*args):
                events.append("train")
                write_json(directory / "training.json", {"best_epoch": 1})
            def evaluate(*args):
                events.append("test")
                write_json(directory / "report.json", {"test": {"ndcg@10": .1}})
            with patch("experiments.train_diffusion", side_effect=train), patch("experiments.final_evaluation", side_effect=evaluate):
                train_run(directory, Path(temp) / "source", self.config, torch.device("cpu"))
                train_run(directory, Path(temp) / "source", self.config, torch.device("cpu"))
            self.assertEqual(events, ["train", "test"])

    def test_prepared_data_reference_resolves_two_stage_links(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "experiment" / "data_source.json", {"path": str(root / "candidates")})
            write_json(root / "candidates" / "data_source.json", {"path": str(root / "data")})
            self.assertEqual(prepared_root(root / "experiment"), root / "data")

    def test_balanced_partition_reml_separates_between_and_within_variation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for partition, scores in enumerate(((.1, .2, .3), (.4, .5, .6), (.7, .8, .9))):
                for seed, score in enumerate(scores):
                    directory = root / "experiments" / "Toys" / "shuffle_partition" / f"p{partition}_s{seed}"
                    write_json(directory / "resolved_config.json", {"partition_seed": partition})
                    write_json(directory / "report.json", {"test": {"ndcg@10": score}})
            result = partition_statistics(root)[0]
            self.assertAlmostEqual(result["grand_mean"], .5)
            self.assertAlmostEqual(result["within_sd_reml"], .1)
            self.assertAlmostEqual(result["partition_sd_reml"] ** 2, .09 - .01 / 3)
            self.assertAlmostEqual(result["total_sd_reml"] ** 2,
                                   result["partition_sd_reml"] ** 2 + result["within_sd_reml"] ** 2)


if __name__ == "__main__":
    unittest.main()
