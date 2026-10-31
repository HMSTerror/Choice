"""SASRec proposer and NewChoice DDiTBlock_UCUR / DDitFinalLayer4Rec denoiser.

Core layers ported from NewChoice/model/transformer.py; no ranking auxiliary.
"""
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
        layer = nn.TransformerEncoderLayer(width, settings["heads"], settings.get("intermediate_size", width * 4),
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


class LayerNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value):
        return F.layer_norm(value.float(), [value.shape[-1]]) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, length, width):
        super().__init__()
        thetas = 1. / (10000 ** (torch.arange(0, width, 2).float() / width))
        frequencies = torch.einsum("i,j->ij", torch.arange(length).float(), thetas)
        angles = torch.cat((frequencies, frequencies), -1)
        self.register_buffer("cos_cached", angles.cos().unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", angles.sin().unsqueeze(0), persistent=False)

    def forward(self, value):
        first, second = value.chunk(2, -1)
        return value * self.cos_cached + torch.cat((-second, first), -1) * self.sin_cached


class PositionwiseFeedForward(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.w_1 = nn.Conv1d(width, width, 1)
        self.w_2 = nn.Conv1d(width, width, 1)
        self.layer_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        output = self.w_2(F.relu(self.w_1(value.transpose(1, 2)))).transpose(1, 2)
        return self.layer_norm(self.dropout(output) + value)


class MultiHeadAttention(nn.Module):
    def __init__(self, width, heads, dropout, head_scale=True):
        super().__init__()
        self.hidden_size, self.num_heads, self.head_scale = width, heads, head_scale
        self.linear_q, self.linear_k, self.linear_v = (nn.Linear(width, width) for _ in range(3))
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, valid_mask):
        split = self.hidden_size // self.num_heads
        q, k, v = (torch.cat(torch.split(layer(value), split, dim=2), dim=0)
                   for layer, value in ((self.linear_q, queries), (self.linear_k, keys), (self.linear_v, keys)))
        scale = split if self.head_scale else self.hidden_size
        logits = torch.bmm(q, k.transpose(1, 2)) / scale ** .5
        valid = valid_mask.repeat(self.num_heads, 1)
        causal = torch.ones(queries.shape[1], keys.shape[1], device=keys.device, dtype=torch.bool).tril()
        allowed = valid[:, None, :] & causal[None]
        weights = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min).softmax(-1)
        weights = weights.masked_fill(~allowed, 0.) * valid[:, :, None]
        output = torch.bmm(self.dropout(weights), v)
        output = torch.cat(torch.split(output, queries.shape[0], dim=0), dim=2)
        return (output + queries) * valid_mask.unsqueeze(-1)


class DDiTBlock_UCUR(nn.Module):
    def __init__(self, width, heads, length, dropout=.1, head_scale=True):
        super().__init__()
        self.position_embedding = RotaryEmbedding(length, width)
        self.emb_dropout = nn.Dropout(dropout)
        self.norm1 = LayerNorm(width)
        self.attn = MultiHeadAttention(width, heads, dropout, head_scale)
        self.norm2 = LayerNorm(width)
        self.mlp = PositionwiseFeedForward(width, dropout)
        self.norm3 = LayerNorm(width)

    def forward(self, hidden, valid_mask):
        inputs = self.emb_dropout(self.position_embedding(hidden)) * valid_mask.unsqueeze(-1)
        attended = self.attn(self.norm1(inputs), inputs, valid_mask)
        return self.norm3(self.mlp(self.norm2(attended))) * valid_mask.unsqueeze(-1)


class DDitFinalLayer4Rec(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.linear = nn.Linear(cond_dim + hidden_size, out_channels)

    def forward(self, noisy, condition):
        return self.linear(torch.cat((noisy, condition.unsqueeze(1)), dim=2))


class TimestepEmbedder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, sigma):
        frequencies = torch.exp(-math.log(10000) * torch.arange(128, device=sigma.device, dtype=torch.float32) / 128)
        angles = sigma[:, None].float() * frequencies[None]
        return self.mlp(torch.cat((angles.cos(), angles.sin()), -1))


def bounded_choice_log_ratio(relative, cap=4., temperature=1.):
    return cap * torch.tanh(relative / (temperature * cap))


class Denoiser(nn.Module):
    def __init__(self, item_count, config):
        super().__init__()
        width, length = config["hidden_size"], config["history_length"]
        settings = config["training"]
        self.item_count = item_count
        self.catalog_count = config.get("catalog_count", item_count)
        self.cap = settings["log_score_cap"]
        self.temperature = settings.get("score_temperature", 1.)
        self.drop_history = 0.2 if config["dataset"] in {"Toys", "Sports"} else 0.1
        self.items = nn.Embedding(item_count + 1, width, padding_idx=item_count)
        nn.init.normal_(self.items.weight, std=settings["embedding_std"])
        with torch.no_grad():
            self.items.weight[item_count].zero_()
        self.sigma_map = TimestepEmbedder(settings.get("condition_size", width))
        self.nonpreference = nn.Embedding(1, width)
        nn.init.normal_(self.nonpreference.weight, std=1.0)
        self.history_blocks = nn.ModuleList([
            DDiTBlock_UCUR(width, settings["heads"], length, settings["dropout"],
                          settings.get("attention_head_scale", True))
            for _ in range(settings.get("history_blocks", 1))])
        self.output_layer = DDitFinalLayer4Rec(width, width, settings.get("condition_size", width) + width)
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
        hidden = self.items(sequence) * valid[..., None]
        for block in self.history_blocks:
            hidden = block(hidden, valid)
        positions = torch.arange(sequence.shape[1], device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(~valid, -1).max(-1).values
        return hidden[torch.arange(len(sequence), device=sequence.device), last.clamp_min(0)] * (last >= 0)[:, None]

    def candidate_mass(self, sequence):
        return .001 + .049 * self.mass_head(self.encode_history(sequence)).sigmoid()

    def raw_scores(self, encoded_history, current, sigma):
        time = F.silu(self.sigma_map(sigma))
        noisy = self.items(current)
        state = self.output_layer(noisy[:, None], torch.cat((encoded_history, time), -1)).squeeze(1)
        logits = state @ self.items.weight[:self.item_count].T
        return logits - (state * noisy).sum(-1, keepdim=True)

    def forward(self, sequence, current, sigma):
        history = self.encode_history(sequence)
        if self.training:
            keep = (torch.rand(len(sequence), 1, device=sequence.device) > self.drop_history)
            history = torch.where(keep, history, self.nonpreference.weight.expand_as(history))
        raw = self.raw_scores(history, current, sigma)
        return bounded_choice_log_ratio(raw, self.cap, self.temperature).scatter(1, current[:, None], 0.0)

    def guided_scores(self, history, current, sigma, strength):
        preferred = self.raw_scores(history, current, sigma)
        shared = self.raw_scores(self.nonpreference.weight.expand_as(history), current, sigma)
        raw = strength * preferred + (1 - strength) * shared
        return bounded_choice_log_ratio(raw, self.cap, self.temperature).scatter(1, current[:, None], 0.0)
