"""Weights survive their appendix consumers but do not accumulate after testing."""
import json
from pathlib import Path
import tempfile
import unittest

from checkpoint_policy import cleanup_completed, prune_run
from data import write_json


class CheckpointPolicyTests(unittest.TestCase):
    def test_pruning_keeps_two_choice_weights_and_artifacts_but_no_old_snapshots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("choice_best.pt", "choice_newst.pt", "candidates.pt", "report.json"):
                (root / name).touch()
            (root / "interrupted_old").mkdir()
            (root / "interrupted_old" / "choice_best.pt").touch()
            prune_run(root, {"method": "choicegrow"})
            self.assertTrue((root / "choice_best.pt").exists())
            self.assertTrue((root / "choice_newst.pt").exists())
            self.assertTrue((root / "candidates.pt").exists())
            self.assertFalse((root / "interrupted_old" / "choice_best.pt").exists())
            prune_run(root, {"method": "prefergrow"})
            self.assertEqual(sorted(path.name for path in root.glob("*.pt")), ["candidates.pt"])

    def test_active_legacy_weights_and_efficiency_inputs_survive_until_consumed(self):
        suite = json.loads((Path(__file__).parent / "paper_suite.json").read_text())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = [{"dataset": "Steam", "arm": arm, "seed": 100, "partition": None}
                    for arm in ("prefergrow", "caser")]
            write_json(root / "experiment_plan.json", plan)
            candidate = root / "candidates" / "Steam" / "seed_100"
            candidate.mkdir(parents=True)
            (candidate / "sasrec_best.pt").touch()
            prefer = root / "experiments" / "Steam" / "prefergrow" / "seed_100"
            caser = root / "experiments" / "Steam" / "caser" / "seed_100"
            for directory, method, weight in ((prefer, "prefergrow", "choice_best.pt"), (caser, "caser", "baseline_best.pt")):
                write_json(directory / "resolved_config.json", {"method": method})
                (directory / weight).touch()
            write_json(root / "parallel" / "fit_Steam_prefergrow_100.done.json", {})
            cleanup_completed(root, suite, ["main", "auxiliary", "efficiency"])
            self.assertTrue((caser / "baseline_best.pt").exists())
            self.assertTrue((prefer / "choice_best.pt").exists())
            self.assertTrue((candidate / "sasrec_best.pt").exists())
            write_json(root / "parallel" / "fit_Steam_caser_100.done.json", {})
            cleanup_completed(root, suite, ["main", "auxiliary", "efficiency"])
            self.assertFalse((caser / "baseline_best.pt").exists())
            self.assertTrue((prefer / "choice_best.pt").exists())
            write_json(root / "efficiency" / "end_to_end_Steam_prefergrow_100.json", {})
            cleanup_completed(root, suite, ["main", "auxiliary", "efficiency"], finished=True)
            self.assertFalse((prefer / "choice_best.pt").exists())
            self.assertFalse((candidate / "sasrec_best.pt").exists())


if __name__ == "__main__":
    unittest.main()
