"""Compare this denoiser to actual NewChoice source without importing its training dependencies.

Usage: python check_newchoice_parity.py --source /path/to/NewChoice
Only Python definitions for the core model and bounded scoring are loaded.
"""
import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models import Denoiser


class AttrDict(dict):
    def __getattr__(self, name):
        return self[name]


def reference_model(source, config, n):
    path = Path(source) / "model" / "transformer.py"
    names = {"LayerNorm", "TimestepEmbedder", "RotaryEmbedding", "PositionwiseFeedForward",
             "MultiHeadAttention", "DDiTBlock_UCUR", "EmbeddingLayer", "DDitFinalLayer4Rec", "SEDD4REC",
             "bounded_choice_log_ratio", "raw_choice_log_ratio", "bounded_personalized_choice_log_ratio"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    namespace = {"torch": torch, "nn": nn, "F": F, "np": np, "math": math}
    exec(compile(tree, str(path), "exec"), namespace)
    settings = config["training"]
    reference_config = AttrDict(graph=AttrDict(is_disliked_item=False),
        training=AttrDict(data="Beauty", nonpreference_user_ratio=.1),
        data=AttrDict(Beauty=AttrDict(item_num=n, seq_len=config["history_length"])),
        model=AttrDict(hidden_size=config["hidden_size"], cond_dim=config["hidden_size"],
            n_blocks=1, n_heads=settings["heads"], dropout=settings["dropout"],
            score_flag=True, score_method="oricosChoice", history_masking=True,
            attention_head_scale=True, embedding_init_std=settings["embedding_std"], scale_by_sigma=False))
    return namespace["SEDD4REC"](reference_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = json.loads((Path(__file__).parent / "config.json").read_text())
    config.update(dataset="Beauty", hidden_size=8, history_length=4)
    config["training"]["dropout"] = .25
    torch.manual_seed(718)
    reference = reference_model(args.source, config, 12)
    torch.manual_seed(718)
    actual = Denoiser(12, config)
    def name(key):
        return key.replace("vocab_embed.embedding.", "items.").replace("nonpreference_user.", "nonpreference.")
    actual_parameters = dict(actual.named_parameters())
    for key, parameter in reference.named_parameters():
        torch.testing.assert_close(parameter, actual_parameters[name(key)], rtol=0, atol=0)
    sequence = torch.tensor([[12, 12, 1, 2], [12, 3, 4, 5], [12, 12, 12, 12]])
    current = torch.tensor([4, 7, 2]);sigma = torch.tensor([.001, .3, 10.])
    for training in (False, True):
        reference.train(training);actual.train(training)
        # Original forward() always drops history; its evaluator uses forward_eval().
        reference.nonpreference_user_ratio = .1 if training else 0.
        torch.manual_seed(17)
        expected = reference(sequence, current[:, None], sigma).squeeze(1)
        torch.manual_seed(17)
        observed = actual(sequence, current, sigma)
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)
        expected.square().sum().backward();observed.square().sum().backward()
        for key, parameter in reference.named_parameters():
            torch.testing.assert_close(parameter.grad, actual_parameters[name(key)].grad, rtol=1e-5, atol=1e-6)
        reference.zero_grad();actual.zero_grad()
        print(f"PASS initialization, forward and gradients: training={training}")
    reference.eval();actual.eval()
    for strength in (0., 1., 2.):
        expected = reference.forward_eval(sequence, current[:, None], sigma, strength).squeeze(1)
        observed = actual.guided_scores(actual.encode_history(sequence), current, sigma, strength)
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)
    print("PASS bounded classifier-free guidance at strengths 0, 1, 2")


if __name__ == "__main__":
    main()
