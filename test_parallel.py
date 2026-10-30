"""Parallel dependency correctness and compact candidate storage contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from data import loader, write_json
from experiments import baseline_search_configs, choice_search_configs
from parallel import build_graph


ROOT = Path(__file__).resolve().parent


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.suite = json.loads((ROOT / "paper_suite.json").read_text())
        self.base = json.loads((ROOT / "config.json").read_text())

    def test_every_formal_job_appears_once_with_its_own_candidate_and_selection_dependencies(self):
        graph = build_graph(self.suite, self.suite["datasets"], self.suite["seeds"], self.suite["groups"], "all")
        ids = [row["id"] for row in graph]
        self.assertEqual(len(ids), len(set(ids)))
        experiments = [row for row in graph if row["kind"] == "experiment"]
        self.assertEqual(len(experiments), 468)
        for row in graph:
            self.assertTrue(set(row["dependencies"]).issubset(ids))
        for row in experiments:
            self.assertIn(f"candidates_{row['dataset']}_{row['seed']}", row["dependencies"])
        for dataset in self.suite["datasets"]:
            selection = next(r for r in graph if r["id"] == f"tune_{dataset}")
            expected_trials = self.suite["tuning"]["trials"][dataset] if self.suite["tuning"]["enabled"] else 0
            self.assertEqual(len([d for d in selection["dependencies"] if "_choice_" in d]), expected_trials)
        completed = set()
        while len(completed) < len(graph):
            ready = {row["id"] for row in graph if set(row["dependencies"]).issubset(completed)} - completed
            self.assertTrue(ready, "Dependency graph contains a cycle")
            completed.update(ready)

    def test_search_proposals_are_reproducible_and_baseline_search_is_independent(self):
        self.suite["tuning"]["enabled"] = True
        self.suite["baselines"]["tuning"] = True
        source = Path("candidates/Toys/seed_100")
        first = choice_search_configs(self.base, self.suite, "Toys", 100, source, False)
        self.assertEqual(first, choice_search_configs(self.base, self.suite, "Toys", 100, source, False))
        self.assertEqual(len(first), 100)
        ddsr = baseline_search_configs(Path("runs/paper"), self.base, self.suite, "Toys", "ddsr", source, False, 100)
        self.assertEqual(len(ddsr), 24)
        self.assertEqual({c["baseline"]["ddsr"]["skip"] for c in ddsr}, set(self.suite["baselines"]["ddsr_skips"]))

    def test_fixed_default_graph_has_no_model_search_tasks(self):
        graph = build_graph(self.suite, self.suite["datasets"], self.suite["seeds"], self.suite["groups"], "all")
        self.assertFalse(any(row["kind"] in {"choice_trial", "baseline_trial"} for row in graph))
        self.assertEqual(len(graph), 528)

    def test_compact_candidate_ids_are_restored_to_int64_at_model_boundary(self):
        values = {"seq": torch.tensor([[3, 0], [0, 1]]), "length": torch.tensor([1, 2]), "target": torch.tensor([1, 2])}
        proposed = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)
        batch = next(iter(loader(values, 2, candidates=proposed)))
        self.assertEqual(batch[-1].dtype, torch.int64)
        torch.testing.assert_close(batch[-1], proposed.long())

    def test_json_replacement_leaves_a_complete_document_and_no_temporary_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            write_json(path, {"state": "running"})
            write_json(path, {"state": "complete"})
            self.assertEqual(json.loads(path.read_text()), {"state": "complete"})
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_vectorized_loader_preserves_old_shuffle_order_across_epochs_and_final_partial_batch(self):
        from torch.utils.data import DataLoader, TensorDataset
        values = {"seq": torch.arange(70).reshape(35, 2), "length": torch.ones(35, dtype=torch.long), "target": torch.arange(35)}
        original = DataLoader(TensorDataset(*values.values()), batch_size=8, shuffle=True,
                              generator=torch.Generator().manual_seed(100))
        vectorized = loader(values, 8, shuffle=True, seed=100)
        for epoch in range(2):
            for expected, actual in zip(original, vectorized):
                for a, b in zip(expected, actual):
                    torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
