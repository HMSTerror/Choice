"""SASRec proposer and the retained NewChoice linear denoiser. No ranking auxiliary."""
import math

import torch
from torch import nn
import torch.nn.functional as F


class SASRec(nn.Module):
    def __init__(self, item_count, config):
        super().__init__()
        width, length = config["hidden_size"], config["history_length"]
        settings = config["sasrec"]
        self.item_count = item_count
        self.items = nn.Embedding(item_count + 1, width, padding_idx=item_count)
        nn.init.normal_(self.items.weight, std=0.02)
        with torch.no_grad():
            self.items.weight[item_count].zero_()
        self.positions = nn.Embedding(length, width)
        layer = nn.TransformerEncoderLayer(width, settings["heads"], width * 4,
                    settings["dropout"], activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, settings["layers"], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(settings["dropout"])

    def encode(self, sequence, lengths):
        # Data is left-padded; the original SASRec encoder expects right padding.
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        indices = (positions[None] + sequence.shape[1] - lengths[:, None]) % sequence.shape[1]
        sequence = sequence.gather(1, indices)
        hidden = self.dropout(self.items(sequence) + self.positions(positions)[None])
        padding = positions[None] >= lengths[:, None]
        causal = torch.ones(sequence.shape[1], sequence.shape[1], device=sequence.device,
                            dtype=torch.bool).triu(1)
        hidden = self.norm(self.encoder(hidden, mask=causal, src_key_padding_mask=padding))
        return hidden[torch.arange(len(sequence), device=sequence.device), lengths - 1]

    def forward(self, sequence, lengths):
        return self.encode(sequence, lengths) @ self.items.weight[:self.item_count].T


class ScaleNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value):
        return F.layer_norm(value.float(), [value.shape[-1]]) * self.weight


class HistoryEncoder(nn.Module):
    def __init__(self, width, length, heads, dropout):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2, self.norm3 = (ScaleNorm(width) for _ in range(3))
        self.query, self.key, self.value = (nn.Linear(width, width) for _ in range(3))
        self.ff1, self.ff2 = nn.Linear(width, width), nn.Linear(width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        frequencies = torch.outer(torch.arange(length), 10000.0 ** (-torch.arange(0, width, 2) / width))
        angles = torch.cat((frequencies, frequencies), -1)
        self.register_buffer("rotary_cos", angles.cos())
        self.register_buffer("rotary_sin", angles.sin())

    def forward(self, hidden, valid):
        batch, length, width = hidden.shape
        first, second = hidden.chunk(2, -1)
        hidden = self.dropout(hidden * self.rotary_cos + torch.cat((-second, first), -1) * self.rotary_sin)
        hidden = hidden * valid[..., None]
        queries = self.norm1(hidden)
        reshape = lambda value: value.reshape(batch, length, self.heads, width // self.heads).transpose(1, 2)
        q, k, v = reshape(self.query(queries)), reshape(self.key(hidden)), reshape(self.value(hidden))
        logits = q @ k.transpose(-1, -2) / math.sqrt(width // self.heads)
        allowed = valid[:, None, None, :] & torch.ones(length, length, device=hidden.device,
                                                       dtype=torch.bool).tril()[None, None]
        weights = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min).softmax(-1)
        weights = weights.masked_fill(~allowed, 0) * valid[:, None, :, None]
        attended = (self.dropout(weights) @ v).transpose(1, 2).reshape(batch, length, width)
        hidden = self.norm2((attended + queries) * valid[..., None])
        hidden = self.ff_norm(hidden + self.dropout(self.ff2(F.relu(self.ff1(hidden)))))
        return self.norm3(hidden) * valid[..., None]


class Denoiser(nn.Module):
    def __init__(self, item_count, config):
        super().__init__()
        width, length = config["hidden_size"], config["history_length"]
        settings = config["training"]
        self.item_count = item_count
        self.catalog_count = config.get("catalog_count", item_count)
        self.cap = settings["log_score_cap"]
        self.drop_history = 0.2 if config["dataset"] in {"Toys", "Sports"} else 0.1
        self.items = nn.Embedding(item_count + 1, width, padding_idx=item_count)
        nn.init.normal_(self.items.weight, std=settings["embedding_std"])
        with torch.no_grad():
            self.items.weight[item_count].zero_()
        self.time = nn.Sequential(nn.Linear(256, width), nn.GELU(), nn.Linear(width, width))
        self.nonpreference = nn.Embedding(1, width)
        nn.init.normal_(self.nonpreference.weight, std=1.0)
        self.history = HistoryEncoder(width, length, settings["heads"], settings["dropout"])
        self.output = nn.Linear(width * 3, width)
        if config.get("method") == "prefergrow" and config.get("prefergrow", {}).get("variant") == "adaptive":
            self.global_destination = nn.Parameter(torch.zeros(item_count))
        if config["destination"].get("candidate_gate") == "dynamic_mass":
            self.mass_head = nn.Linear(width, 1)
            nn.init.zeros_(self.mass_head.weight)
            m = config["destination"]["candidate_mass"]
            p = min(max((m - .001) / .049, 1e-4), 1 - 1e-4)
            nn.init.constant_(self.mass_head.bias, math.log(p / (1 - p)))

    def encode_history(self, sequence):
        valid = sequence < self.catalog_count
        sequence = sequence.masked_fill(~valid, self.item_count)
        hidden = self.history(self.items(sequence) * valid[..., None], valid)
        # Preprocessing enforces nonempty, left-padded histories.
        return hidden[:, -1]

    def candidate_mass(self, sequence):
        return .001 + .049 * self.mass_head(self.encode_history(sequence)).sigmoid()

    def raw_scores(self, encoded_history, current, sigma):
        frequencies = torch.exp(-math.log(10000) * torch.arange(128, device=sigma.device) / 128)
        angles = sigma[:, None] * frequencies[None]
        time = F.silu(self.time(torch.cat((angles.cos(), angles.sin()), -1)))
        noisy = self.items(current)
        state = self.output(torch.cat((noisy, encoded_history, time), -1))
        logits = state @ self.items.weight[:self.item_count].T
        return logits - (state * noisy).sum(-1, keepdim=True)

    def forward(self, sequence, current, sigma):
        history = self.encode_history(sequence)
        if self.training:
            keep = (torch.rand(len(sequence), 1, device=sequence.device) > self.drop_history)
            history = torch.where(keep, history, self.nonpreference.weight.expand_as(history))
        raw = self.raw_scores(history, current, sigma)
        return (self.cap * torch.tanh(raw / self.cap)).scatter(1, current[:, None], 0.0)

    def guided_scores(self, history, current, sigma, strength):
        preferred = self.raw_scores(history, current, sigma)
        shared = self.raw_scores(self.nonpreference.weight.expand_as(history), current, sigma)
        raw = strength * preferred + (1 - strength) * shared
        return (self.cap * torch.tanh(raw / self.cap)).scatter(1, current[:, None], 0.0)
