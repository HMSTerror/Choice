"""PyTorch port of FeiSun/BERT4Rec's model, masking and sampled evaluation.

Reference: official commit 615eaf2004abecda487a38d5b0c72f3dcfcae5b3.
Protocol boundaries are documented in docs/BERT4REC_ALIGNMENT.md.
Note: .agents/notes/implemented/architecture/2026-10-30-paper-suite.md
"""
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import BatchedColumns, batched_identity, load_split, metadata, write_json


PROTOCOL = "bert_popularity_100"


def gelu(value):
    # modeling.py uses BERT's tanh approximation, not the erf implementation.
    return .5 * value * (1 + torch.tanh(math.sqrt(2 / math.pi) * (value + .044715 * value.pow(3))))


class BertBlock(nn.Module):
    def __init__(self, width, heads, hidden_dropout, attention_dropout):
        super().__init__()
        self.heads = heads
        self.query, self.key, self.value = (nn.Linear(width, width) for _ in range(3))
        self.attention_output = nn.Linear(width, width)
        self.attention_norm = nn.LayerNorm(width, eps=1e-12)
        self.intermediate = nn.Linear(width, 4 * width)
        self.output = nn.Linear(4 * width, width)
        self.output_norm = nn.LayerNorm(width, eps=1e-12)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, hidden, valid):
        b, length, width = hidden.shape
        def split(layer):
            return layer(hidden).reshape(b, length, self.heads, width // self.heads).transpose(1, 2)
        query, key, value = split(self.query), split(self.key), split(self.value)
        scores = query @ key.transpose(-1, -2) / math.sqrt(width // self.heads)
        scores = scores + (~valid[:, None, None, :]).to(scores.dtype) * -10000.
        attention = self.attention_dropout(scores.softmax(-1)) @ value
        attention = attention.transpose(1, 2).reshape(b, length, width)
        hidden = self.attention_norm(hidden + self.hidden_dropout(self.attention_output(attention)))
        # No dropout between intermediate GELU and output dense in official BERT.
        output = self.output(gelu(self.intermediate(hidden)))
        return self.output_norm(hidden + self.hidden_dropout(output))


def mask_tokens(tokens, mask_id, probability, max_predictions, rng):
    """gen_data_fin.py: fixed mask count, shuffled positions, mask_prob=1.0."""
    positions = list(range(len(tokens)))
    rng.shuffle(positions)
    count = min(max_predictions, max(1, int(round(len(tokens) * probability))))
    selected = sorted(positions[:count])
    corrupted = list(tokens)
    for index in selected:
        corrupted[index] = mask_id
    # Preserve the author's RNG consumption after the shuffled selection.
    for _ in selected:
        rng.random()
    return corrupted, selected


class BERT4Rec(nn.Module):
    def __init__(self, n, config):
        super().__init__()
        self.n, self.mask_id = n, n + 1
        settings = config["baseline"]
        width = config["hidden_size"]
        self.max_length = settings.get("bert_max_length", 50)
        self.mask_probability = settings.get("bert_mask_probability", .6)
        self.max_predictions = settings.get("bert_max_predictions", 30)
        self.mask_rng = random.Random(config["seed"])
        # Author vocabulary includes item IDs plus PAD/MASK/NO_USE and unused ID 0.
        # Item IDs remain 0..n-1 here; all four special rows participate in MLM CE.
        self.items = nn.Embedding(n + 4, width)
        self.positions = nn.Embedding(self.max_length, width)
        self.token_types = nn.Embedding(2, width)
        self.norm = nn.LayerNorm(width, eps=1e-12)
        self.dropout = nn.Dropout(settings["dropout"])
        self.blocks = nn.ModuleList(BertBlock(width, settings.get("bert_heads", 2),
                         settings["dropout"], settings.get("bert_attention_dropout", .2))
                         for _ in range(settings.get("bert_layers", 2)))
        self.prediction_dense = nn.Linear(width, width)
        self.prediction_norm = nn.LayerNorm(width, eps=1e-12)
        self.output_bias = nn.Parameter(torch.zeros(n + 4))
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=.02, a=-.04, b=.04)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def encode(self, sequence):
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        hidden = self.items(sequence) + self.positions(positions) + self.token_types.weight[0]
        hidden = self.dropout(self.norm(hidden))
        for block in self.blocks:
            hidden = block(hidden, sequence != self.n)
        return self.prediction_norm(gelu(self.prediction_dense(hidden)))

    def clean_tokens(self, sequence, lengths, target):
        rows = []
        for history, length, item in zip(sequence.tolist(), lengths.tolist(), target.tolist()):
            rows.append((history[-length:] + [item])[-self.max_length:])
        return rows

    def masked_loss(self, inputs, labels):
        selected = labels != -100
        logits = self.encode(inputs)[selected] @ self.items.weight.T + self.output_bias
        return F.cross_entropy(logits, labels[selected])

    def loss(self, sequence, lengths, target):
        # Public single-batch API; formal training uses fixed duplicated masks below.
        rows = self.clean_tokens(sequence, lengths, target)
        inputs, labels = masked_examples(rows, self, self.mask_rng)
        return self.masked_loss(inputs.to(sequence.device), labels.to(sequence.device))

    def next_state(self, sequence, lengths):
        # The author's inputs start at position 0 and are padded on the right.
        sequence = sequence[:, -(self.max_length - 1):]
        lengths = lengths.clamp_max(sequence.shape[1])
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        indices = (positions[None] + sequence.shape[1] - lengths[:, None]) % sequence.shape[1]
        history = sequence.gather(1, indices)
        inputs = torch.full((len(sequence), self.max_length), self.n,
                            dtype=torch.long, device=sequence.device)
        inputs[:, :sequence.shape[1]] = history
        inputs.scatter_(1, lengths[:, None], self.mask_id)
        return self.encode(inputs)[torch.arange(len(sequence), device=sequence.device), lengths]

    def candidate_scores(self, sequence, lengths, candidates):
        state = self.next_state(sequence, lengths)
        return (state[:, None] * self.items(candidates)).sum(-1) + self.output_bias[candidates]

    def forward(self, sequence, lengths):
        # Diagnostic API. Formal validation/test call candidate_scores, never this.
        return self.next_state(sequence, lengths) @ self.items.weight[:self.n].T + self.output_bias[:self.n]


def masked_examples(rows, model, rng, force_last=False):
    inputs = torch.full((len(rows), model.max_length), model.n, dtype=torch.long)
    labels = torch.full_like(inputs, -100)
    for index, tokens in enumerate(rows):
        if force_last:
            corrupted, selected = list(tokens), [len(tokens) - 1]
            corrupted[-1] = model.mask_id
        else:
            corrupted, selected = mask_tokens(tokens, model.mask_id, model.mask_probability,
                                               model.max_predictions, rng)
        inputs[index, :len(tokens)] = torch.tensor(corrupted)
        labels[index, selected] = torch.tensor([tokens[p] for p in selected])
    return inputs, labels


def training_loader(values, model, config):
    """Ten fixed masked copies and one mask-last copy, prepared from TRAIN only."""
    rows = model.clean_tokens(*(values[key] for key in ("seq", "length", "target")))
    rng = random.Random(config["seed"])
    copies = [masked_examples(rows, model, rng)
              for _ in range(config["baseline"].get("bert_dupe_factor", 10))]
    copies.append(masked_examples(rows, model, rng, force_last=True))
    inputs, labels = (torch.cat([pair[column] for pair in copies]) for column in (0, 1))
    return DataLoader(BatchedColumns((inputs, labels)), batch_size=config["training"]["batch_size"],
                      shuffle=True, collate_fn=batched_identity,
                      generator=torch.Generator().manual_seed(config["seed"]))


def popularity_counts(training, item_count):
    items = torch.cat((training["seq"].reshape(-1), training["target"]))
    return torch.bincount(items[items < item_count], minlength=item_count).numpy()


def sample_candidates(values, counts, negative_count=100, seed=12345):
    """Author's rejection sampler: weighted draws without replacement, unseen IDs."""
    ids = np.flatnonzero(counts)
    probability = counts[ids].astype(np.float64)
    probability /= probability.sum()
    rng = np.random.RandomState(seed)
    candidates = np.empty((len(values["target"]), negative_count + 1), dtype=np.int64)
    for index, (history, target) in enumerate(zip(values["seq"].tolist(), values["target"].tolist())):
        excluded = set(history) | {target}
        if len(ids) - int(np.isin(ids, list(excluded)).sum()) < negative_count:
            raise ValueError("Not enough unseen items with training frequency for BERT evaluation")
        chosen = [target]
        while len(chosen) < negative_count + 1:
            sampled = rng.choice(ids, min(negative_count + 1, len(ids)), replace=False, p=probability)
            chosen.extend(int(item) for item in sampled if item not in excluded and item not in chosen)
        candidates[index] = chosen[:negative_count + 1]
    return torch.from_numpy(candidates)


def evaluation_candidates(output, values, config, split):
    # Never estimate popularity from validation/test targets. The old author
    # script builds its vocabulary counter from user_test_data; see disclosure.
    count = config["baseline"].get("bert_eval_negatives", 100)
    seed = config["baseline"].get("bert_eval_seed", 12345)
    counts = popularity_counts(load_split(output, "train"), metadata(output)["item_count"])
    candidates = sample_candidates(values, counts, count, seed)
    np.save(Path(output) / f"bert_{split}_candidates.npy", candidates.numpy())
    write_json(Path(output) / f"bert_{split}_evaluation.json", {
        "protocol": PROTOCOL, "negative_count": count, "candidate_seed": seed,
        "popularity_source": "train_history_and_target_occurrences",
        "excluded": "provided_history_and_target", "split": split,
    })
    return candidates


def ranking_totals(scores):
    # Match EvalHooks' NumPy rank calculation, including its tie convention.
    rank = (-scores.detach().float().cpu().numpy()).argsort(axis=1).argsort(axis=1)[:, 0]
    return {key: float(value) for cutoff in (5, 10) for key, value in (
        (f"hr@{cutoff}", (rank < cutoff).sum()),
        (f"ndcg@{cutoff}", ((rank < cutoff) / np.log2(rank + 2)).sum()))}
