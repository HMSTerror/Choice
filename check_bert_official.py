"""Run the author's pure Python masking and EvalHooks against the local port."""
import argparse
from collections import namedtuple
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

from bert4rec import mask_tokens, sample_candidates, ranking_totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="FeiSun/BERT4Rec checkout at 615eaf2")
    args = parser.parse_args()
    source = Path(args.source)
    # Extract the functions independently: gen_data_fin.py as a whole is Python 2.
    text = (source / "gen_data_fin.py").read_text(encoding="utf-8")
    function = "def create_masked_lm_predictions(" + text.split("def create_masked_lm_predictions(", 1)[1].split("\ndef gen_samples(", 1)[0]
    namespace = {"MaskedLmInstance": namedtuple("MaskedLmInstance", ["index", "label"])}
    exec(compile(function, str(source / "gen_data_fin.py"), "exec"), namespace)
    official = namespace["create_masked_lm_predictions"]
    left, right = random.Random(100), random.Random(100)
    for length in (2, 3, 11, 50, 8, 11):
        tokens = list(range(length))
        expected, selected, labels = official(tokens, .6, 30, tokens, left, 1.)
        actual, positions = mask_tokens(tokens, 999, .6, 30, right)
        assert positions == selected
        assert actual == [999 if item == "[MASK]" else item for item in expected]
        assert [tokens[index] for index in positions] == labels
    print("PASS official create_masked_lm_predictions: counts, positions, MASK inputs, RNG progression")
    text = (source / "run.py").read_text(encoding="utf-8")
    function = "    def after_run(" + text.split("    def after_run(", 1)[1].split("\ndef model_fn_builder(", 1)[0]
    namespace = {"np": np, "sys": sys, "FLAGS": SimpleNamespace(max_predictions_per_seq=1, use_pop_random=True)}
    function = "\n".join(line[4:] if line.startswith("    ") else line for line in function.splitlines())
    exec(compile(function, str(source / "run.py"), "exec"), namespace)
    n = 150
    histories = np.array([[n, 1, 2], [n, 4, 5], [n, 7, 8]])
    targets = np.array([3, 6, 9])
    counts = np.arange(1, n + 1)
    # Official IDs begin at 1; assign ID 0 no mass in this fixture.
    counts[0] = 0
    values = {"seq": torch.tensor(histories), "target": torch.tensor(targets)}
    candidates = sample_candidates(values, counts)
    full_scores = np.random.RandomState(22).normal(size=(len(targets), n + 2))
    full_scores[0, targets[0]] = 100.
    actual = ranking_totals(torch.tensor(full_scores).gather(1, candidates))
    state = SimpleNamespace(valid_user=0., hit_1=0., hit_5=0., hit_10=0.,
                            ndcg_1=0., ndcg_5=0., ndcg_10=0., ap=0., vocab=True,
                            ids=np.flatnonzero(counts), probability=counts[counts > 0] / counts.sum(),
                            user_history={"user_" + str(i): [[]] for i in range(len(targets))})
    observed = []
    class Scores(np.ndarray):
        def __getitem__(self, key):
            if isinstance(key, list) and len(key) == 101:
                observed.append(key)
            return super().__getitem__(key)
    run_values = SimpleNamespace(results=(full_scores.view(Scores), histories, targets[:, None], np.arange(len(targets))[:, None]))
    np.random.seed(12345)
    namespace["after_run"](state, None, run_values)
    assert candidates.tolist() == observed
    for cutoff in (5, 10):
        np.testing.assert_allclose(actual[f"hr@{cutoff}"], getattr(state, f"hit_{cutoff}"))
        np.testing.assert_allclose(actual[f"ndcg@{cutoff}"], getattr(state, f"ndcg_{cutoff}"))
    print("PASS official EvalHooks.after_run: popularity sampler and HR/NDCG candidate ranks")


if __name__ == "__main__":
    main()
