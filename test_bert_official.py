"""Regression checks for the BERT4Rec author protocol and its split boundaries."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from baselines import BERT4Rec, evaluate_baseline
from bert4rec import (training_loader, sample_candidates, popularity_counts,
                      evaluation_candidates, ranking_totals)
from data import write_json
from experiments import aggregate_rows, arm_config, config_for, train_run


class BertOfficialTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def config(self):
        root = Path(__file__).parent
        base = json.loads((root / "config.json").read_text())
        suite = json.loads((root / "paper_suite.json").read_text())
        config = arm_config(config_for(base, "Sports", 100), "bert4rec", suite)
        config["hidden_size"] = 8
        config["history_length"] = 40
        config["baseline"]["bert_mask_probability"] = 1.
        config["baseline"]["bert_max_predictions"] = 50
        return config

    def test_author_run_script_masks_every_selected_item(self):
        torch.manual_seed(1)
        model = BERT4Rec(60, self.config())
        sequence = torch.arange(40)[None]
        captured = []
        encode = model.encode
        def capture(value):
            captured.append(value.clone())
            return encode(value)
        with patch.object(model, "encode", side_effect=capture):
            model.loss(sequence, torch.tensor([40]), torch.tensor([40]))
        # run_beauty.sh sets mask_prob=1.0, rather than BERT's 80/10/10.
        self.assertTrue(bool((captured[0][0, :41] == model.mask_id).all()))

    def test_exact_mask_count_and_fixed_training_copies(self):
        config = self.config()
        config["baseline"].update(bert_mask_probability=.6, bert_dupe_factor=10)
        model = BERT4Rec(60, config)
        values = {"seq": torch.tensor([[60, 60, 1, 2], [60, 3, 4, 5]]),
                  "length": torch.tensor([2, 3]), "target": torch.tensor([6, 7])}
        batches = training_loader(values, model, config)
        inputs, labels = batches.dataset.columns
        self.assertEqual(len(inputs), 22)
        self.assertTrue(bool((inputs[labels != -100] == model.mask_id).all()))
        self.assertEqual((labels[:20] != -100).sum(-1).tolist(), [2, 2] * 10)
        self.assertEqual((labels[-2:] != -100).sum(-1).tolist(), [1, 1])
        self.assertEqual(labels[-2, 2].item(), 6)
        self.assertEqual(labels[-1, 3].item(), 7)
        repeated = training_loader(values, model, config).dataset.columns
        for before, after in zip((inputs, labels), repeated):
            torch.testing.assert_close(before, after)

    def test_candidate_scoring_right_padding_and_no_full_catalog_evaluation(self):
        model = BERT4Rec(60, self.config()).eval()
        sequence = torch.tensor([[60, 60, 1, 2], [60, 3, 4, 5]])
        lengths = torch.tensor([2, 3])
        candidates = torch.tensor([[6, 8, 9, 10], [7, 9, 10, 11]])
        captured = []
        encode = model.encode
        with patch.object(model, "encode", side_effect=lambda x: (captured.append(x.clone()), encode(x))[1]):
            subset = model.candidate_scores(sequence, lengths, candidates)
        self.assertEqual(captured[0][0, :4].tolist(), [1, 2, 61, 60])
        self.assertEqual(captured[0][1, :5].tolist(), [3, 4, 5, 61, 60])
        torch.testing.assert_close(subset, model(sequence, lengths).gather(1, candidates))
        values = {"seq": sequence, "length": lengths, "target": candidates[:, 0]}
        with patch.object(model, "forward", side_effect=AssertionError("Full-catalog scoring called")):
            report = evaluate_baseline(model, values, self.config(), torch.device("cpu"), "test", 3, candidates)
        self.assertEqual(report["streams"], 1)
        self.assertEqual(report["evaluation_protocol"], "bert_popularity_100")
        self.assertEqual(report["negative_count"], 3)

    def test_popularity_candidates_unique_exclude_history_and_are_fixed(self):
        values = {"seq": torch.tensor([[120, 1, 2], [120, 2, 3]]),
                  "target": torch.tensor([4, 5])}
        counts = np.arange(1, 121)
        actual = sample_candidates(values, counts)
        torch.testing.assert_close(actual, sample_candidates(values, counts))
        self.assertEqual(actual.shape, (2, 101))
        self.assertEqual(actual[:, 0].tolist(), [4, 5])
        for row, history in zip(actual.tolist(), values["seq"].tolist()):
            self.assertEqual(len(set(row)), 101)
            self.assertFalse(set(row[1:]) & set(history))
        counts[110:] = 0
        actual = sample_candidates(values, counts)
        self.assertLess(int(actual[:, 1:].max()), 110)

    def test_evaluation_popularity_reads_only_training_split(self):
        config = self.config()
        config["baseline"]["bert_eval_negatives"] = 3
        training = {"seq": torch.tensor([[8, 0, 1], [2, 3, 4]]),
                    "target": torch.tensor([5, 6])}
        evaluation = {"seq": torch.tensor([[8, 0, 1]]), "target": torch.tensor([7])}
        with tempfile.TemporaryDirectory() as temp, \
             patch("bert4rec.load_split", return_value=training) as read, \
             patch("bert4rec.metadata", return_value={"item_count": 8}):
            candidates = evaluation_candidates(Path(temp), evaluation, config, "val")
            read.assert_called_once_with(Path(temp), "train")
            self.assertTrue(set(candidates[0, 1:].tolist()).issubset({2, 3, 4, 5, 6}))
        counts = popularity_counts(training, 8)
        self.assertEqual(counts.tolist(), [1, 1, 1, 1, 1, 1, 1, 0])

    def test_ranking_uses_author_candidate_rank(self):
        totals = ranking_totals(torch.tensor([[3., 2., 1.], [1., 3., 2.]]))
        self.assertEqual(totals["hr@5"], 2.)
        self.assertAlmostEqual(totals["ndcg@5"], 1.5)

    def test_summary_keeps_evaluation_protocols_separate(self):
        row = {"dataset": "Sports", "experiment": "bert4rec", "report": "report.json",
               **{key: .1 for key in ("hr@5", "hr@10", "ndcg@5", "ndcg@10")}}
        groups = aggregate_rows([row, {**row, "evaluation_protocol": "bert_popularity_100"}])
        self.assertEqual(len(groups), 2)
        self.assertEqual([group["runs"] for group in groups], [1, 1])

    def test_formal_train_tests_after_selection_and_saves_no_checkpoint(self):
        config = self.config()
        config.update(history_length=3, smoke=True)
        config["baseline"].update(bert_max_length=5, bert_max_predictions=3,
                                   bert_dupe_factor=1, bert_eval_negatives=3)
        config["training"].update(epochs=1, validation_interval=1, batch_size=4, eval_batch_size=4)
        values = {"seq": np.array([[8, 0, 1], [2, 3, 4], [8, 5, 6], [8, 1, 2]]),
                  "length": np.array([2, 3, 2, 2]), "target": np.array([2, 5, 7, 4])}
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source", Path(temp) / "result"
            (source / "prepared").mkdir(parents=True)
            for split in ("train", "val", "test"):
                np.savez(source / "prepared" / f"{split}.npz", **values)
            write_json(source / "prepared" / "metadata.json", {"item_count": 8})
            with patch("baselines.append_log"):
                train_run(output, source, config, torch.device("cpu"))
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["test"]["evaluation_protocol"], "bert_popularity_100")
            self.assertEqual(list(output.glob("*.pt")), [])
            self.assertTrue((output / "bert_test_candidates.npy").exists())


if __name__ == "__main__":
    unittest.main()
