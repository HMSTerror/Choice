"""Distinct baseline formulations under the common next-item split/evaluator.

Sources and protocol adaptations are recorded in EXPERIMENTS.md. These are
local adapters, not wrappers that require another checkout at runtime.
"""
import math
from copy import deepcopy
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F

from data import loader, load_split, metadata, write_json
from models import SASRec
from train import (append_log, better, evaluation_rng, evaluate_proposer,
                   ranking_totals, seed_all, training_epochs)


BASELINES = ("sasrec", "caser", "gru4rec", "bert4rec", "preferdiff", "diffrec", "ddsr")


def time_embedding(t, width):
    frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2, device=t.device) / (width // 2))
    angles = t.float()[:, None] * frequencies
    return F.pad(torch.cat((angles.cos(), angles.sin()), -1), (0, width % 2))


def right_pad(sequence, lengths):
    positions = torch.arange(sequence.shape[1], device=sequence.device)
    indices = (positions[None] + sequence.shape[1] - lengths[:, None]) % sequence.shape[1]
    return sequence.gather(1, indices)


def sampled_binary_cross_entropy(state, item_embeddings, targets, negative_count):
    item_count = item_embeddings.shape[0]
    if item_count < 2 or negative_count < 1:
        raise ValueError("sampled BCE requires at least two items and one negative")
    negative_ids = torch.randint(item_count - 1, (len(targets), negative_count), device=targets.device)
    negative_ids += negative_ids >= targets[:, None]
    positive_logits = (state * item_embeddings[targets]).sum(-1)
    negative_logits = (state[:, None] * item_embeddings[negative_ids]).sum(-1)
    positive_loss = F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits))
    negative_loss = F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
    return positive_loss + negative_loss


def make_baseline_optimizer(model, config):
    settings = config["training"]
    name = settings.get("optimizer", "adam" if config["method"] in {"gru4rec", "caser"} else "adamw")
    decay = settings.get("weight_decay", 0.)
    epsilon = settings.get("optimizer_epsilon", 1e-8)
    if name == "adagrad":
        return PaperAdagrad(model.parameters(), settings["learning_rate"], epsilon)
    if name == "bert_adam":
        groups = [{"params": [p for name, p in model.named_parameters()
                              if p.requires_grad and (p.ndim == 1 or "norm" in name.lower())], "weight_decay": 0.},
                  {"params": [p for name, p in model.named_parameters()
                              if p.requires_grad and p.ndim > 1 and "norm" not in name.lower()], "weight_decay": decay}]
        return BertAdam(groups, settings["learning_rate"], epsilon)
    optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[name]
    return optimizer_type(model.parameters(), lr=settings["learning_rate"], weight_decay=decay, eps=epsilon)


class PaperAdagrad(torch.optim.Optimizer):
    """Original GRU4Rec places epsilon inside sqrt(accumulator + epsilon)."""
    def __init__(self, parameters, lr, eps):
        super().__init__(parameters, dict(lr=lr, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["sum"] = torch.zeros_like(parameter)
                state["sum"].addcmul_(parameter.grad, parameter.grad)
                denominator = (state["sum"] + group["eps"]).sqrt()
                parameter.addcdiv_(parameter.grad, denominator, value=-group["lr"])


class BertAdam(torch.optim.Optimizer):
    """BERT's uncorrected Adam moments with decoupled weight decay."""
    def __init__(self, parameters, lr, eps):
        super().__init__(parameters, dict(lr=lr, eps=eps, weight_decay=0.))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state.update(m=torch.zeros_like(parameter), v=torch.zeros_like(parameter))
                state["m"].mul_(.9).add_(parameter.grad, alpha=.1)
                state["v"].mul_(.999).addcmul_(parameter.grad, parameter.grad, value=.001)
                update = state["m"] / (state["v"].sqrt() + group["eps"])
                update = update + group["weight_decay"] * parameter
                parameter.add_(update, alpha=-group["lr"])


def baseline_learning_rate(settings, step):
    rate = settings["learning_rate"]
    if not settings.get("decay_steps"):
        return rate
    warmup = settings.get("warmup_steps", 0)
    if step < warmup:
        return rate * step / warmup
    return rate * max(0., 1 - step / settings["decay_steps"])


def top1_loss(scores):
    """RSC15 TOP1 with in-batch targets, including the author's diagonal correction."""
    diagonal = scores.diagonal()
    return ((scores - diagonal[:, None]).sigmoid() + scores.square().sigmoid()).mean() - diagonal.square().sigmoid().mean() / len(scores)


class GRU4Rec(nn.Module):
    def __init__(self, n, config):
        super().__init__()
        d = config["hidden_size"]
        self.n = n
        self.loss_type = config["baseline"].get("gru_loss", "bce")
        self.items = nn.Embedding(n + 1, d, padding_idx=n)
        self.dropout = nn.Dropout(config["baseline"]["dropout"])
        if self.loss_type == "top1":
            # Separate one-hot input matrices for the three original GRU gates.
            self.reset_items = nn.Embedding(n + 1, d, padding_idx=n)
            self.update_items = nn.Embedding(n + 1, d, padding_idx=n)
            self.recurrent = nn.Linear(d, d)
            self.reset = nn.Linear(d, d)
            self.update = nn.Linear(d, d)
            self.output = nn.Linear(d, n)
            for parameter in self.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
                else:
                    nn.init.zeros_(parameter)
            for embedding in (self.items, self.reset_items, self.update_items):
                with torch.no_grad():
                    embedding.weight[n].zero_()
        else:
            self.gru = nn.GRU(d, d, batch_first=True)
        self.negative_count = config["baseline"]["negative_samples"]

    def encode(self, sequence, lengths):
        if self.loss_type == "top1":
            sequence = right_pad(sequence, lengths)
            state = self.items.weight.new_zeros(len(sequence), self.items.embedding_dim)
            x, r, z = self.items(sequence), self.reset_items(sequence), self.update_items(sequence)
            for index in range(sequence.shape[1]):
                reset = (r[:, index] + self.reset(state)).sigmoid()
                update = (z[:, index] + self.update(state)).sigmoid()
                candidate = (x[:, index] + self.recurrent(state * reset)).tanh()
                hidden = self.dropout((1 - update) * state + update * candidate)
                state = torch.where((index < lengths)[:, None], hidden, state)
            return state
        embedded = self.dropout(self.items(right_pad(sequence, lengths)))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.gru(packed)
        return state[-1]

    def forward(self, sequence, lengths):
        state = self.encode(sequence, lengths)
        if self.loss_type == "top1":
            return self.output(state).tanh()
        return state @ self.items.weight[:self.n].T

    def loss(self, sequence, lengths, target):
        if self.loss_type == "top1":
            state = self.encode(sequence, lengths)
            logits = F.linear(state, self.output.weight[target], self.output.bias[target]).tanh()
            return top1_loss(logits)
        return sampled_binary_cross_entropy(self.encode(sequence, lengths), self.items.weight[:self.n],
                                           target, self.negative_count)


class Caser(nn.Module):
    def __init__(self, n, config):
        super().__init__()
        self.n = n
        d = config["hidden_size"]
        length = min(config["history_length"], config["baseline"].get("caser_history_length", config["history_length"]))
        self.length = length
        filters = config["baseline"]["caser_filters"]
        self.items = nn.Embedding(n + 1, d, padding_idx=n)
        vertical_filters = config["baseline"].get("caser_vertical_filters", filters)
        self.vertical = nn.Conv2d(1, vertical_filters, (length, 1))
        self.horizontal = nn.ModuleList(nn.Conv2d(1, filters, (h, d)) for h in range(1, length + 1))
        self.dropout = nn.Dropout(config["baseline"]["dropout"])
        self.hidden = nn.Linear(vertical_filters * d + filters * length, d)
        self.output = nn.Embedding(n, d)
        self.bias = nn.Embedding(n, 1)
        self.negative_count = config["baseline"]["negative_samples"]

    def encode(self, sequence, lengths):
        sequence = sequence[:, -self.length:]
        lengths = lengths.clamp_max(self.length)
        sequence = right_pad(sequence, lengths)
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        valid = positions[None, :] < lengths[:, None]
        embedded = (self.items(sequence) * valid[..., None]).unsqueeze(1)
        vertical = F.relu(self.vertical(embedded)).flatten(1)
        horizontal = [F.relu(conv(embedded)).flatten(2).amax(-1) for conv in self.horizontal]
        return F.relu(self.hidden(self.dropout(torch.cat((vertical, *horizontal), -1))))

    def forward(self, sequence, lengths):
        state = self.encode(sequence, lengths)
        return state @ self.output.weight.T + self.bias.weight.T

    def loss(self, sequence, lengths, target):
        state = self.encode(sequence, lengths)
        negatives = torch.randint(self.n - 1, (len(target), self.negative_count), device=target.device)
        negatives += negatives >= target[:, None]
        positive = (state * self.output(target)).sum(-1) + self.bias(target).squeeze(-1)
        negative = (state[:, None] * self.output(negatives)).sum(-1) + self.bias(negatives).squeeze(-1)
        return F.softplus(-positive).mean() + F.softplus(negative).mean()


class BERT4Rec(nn.Module):
    def __init__(self, n, config):
        super().__init__()
        self.n, self.mask_id = n, n + 1
        d = config["hidden_size"]
        settings = config["baseline"]
        self.mask_probability = settings.get("bert_mask_probability", .2)
        self.items = nn.Embedding(n + 2, d, padding_idx=n)
        self.positions = nn.Embedding(config["history_length"] + 1, d)
        layer = nn.TransformerEncoderLayer(d, settings.get("bert_heads", 2), 4 * d, settings["dropout"],
                    activation="gelu", batch_first=True, norm_first=False, layer_norm_eps=1e-12)
        layer.self_attn.dropout = settings.get("bert_attention_dropout", settings["dropout"])
        self.encoder = nn.TransformerEncoder(layer, settings.get("bert_layers", 1), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d, eps=1e-12)
        self.dropout = nn.Dropout(settings["dropout"])
        self.prediction = (nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d, eps=1e-12))
                           if settings.get("bert_prediction_transform") else nn.Identity())
        self.output_bias = nn.Parameter(torch.zeros(n))
        self.apply(self.initialize)

    def initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def encode(self, sequence):
        hidden = self.items(sequence) + self.positions(torch.arange(sequence.shape[1], device=sequence.device))
        hidden = self.dropout(self.norm(hidden))
        return self.prediction(self.encoder(hidden, src_key_padding_mask=sequence == self.n))

    def loss(self, sequence, lengths, target):
        clean = torch.cat((sequence, target[:, None]), -1)
        selected = (torch.rand_like(clean.float()) < self.mask_probability) & (clean < self.n)
        # Ensure every history supplies supervision, including very short ones.
        selected[:, -1] |= ~selected.any(-1)
        draw = torch.rand_like(clean.float())
        corrupt = clean.clone()
        corrupt[selected & (draw < .8)] = self.mask_id
        random_mask = selected & (draw >= .8) & (draw < .9)
        corrupt[random_mask] = torch.randint(self.n, (int(random_mask.sum()),), device=sequence.device)
        logits = self.encode(corrupt)[selected] @ self.items.weight[:self.n].T + self.output_bias
        return F.cross_entropy(logits, clean[selected])

    def forward(self, sequence, lengths):
        sequence = torch.cat((sequence, torch.full_like(sequence[:, :1], self.mask_id)), -1)
        return self.encode(sequence)[:, -1] @ self.items.weight[:self.n].T + self.output_bias


class Gaussian(nn.Module):
    def __init__(self, beta):
        super().__init__()
        beta = beta.double()
        alpha = 1 - beta
        abar = alpha.cumprod(0)
        previous = F.pad(abar[:-1], (1, 0), value=1.)
        self.register_buffer("abar", abar.float())
        self.register_buffer("posterior1", (beta * previous.sqrt() / (1 - abar)).float())
        self.register_buffer("posterior2", ((1 - previous) * alpha.sqrt() / (1 - abar)).float())

    def noise(self, x, t):
        a = self.abar[t, None]
        return a.sqrt() * x + (1 - a).sqrt() * torch.randn_like(x)


class PreferDiff(nn.Module):
    def __init__(self, n, config, artifact):
        super().__init__()
        self.n = n
        self.settings = config["baseline"]["preferdiff"]
        d = config["hidden_size"]
        history_config = deepcopy(config)
        history_config["sasrec"].update(dropout=.1, heads=2, layers=1, intermediate_size=d)
        self.history = SASRec(n, history_config)
        features = F.pad(artifact["features"], (0, 0, 0, 1))
        if features.shape[1] != d:
            raise ValueError(f"PreferDiff features have width {features.shape[1]}, expected {d}; rebuild semantic artifacts")
        self.history.items = nn.Embedding.from_pretrained(features, freeze=True, padding_idx=n)
        self.none = nn.Parameter(torch.randn(1, d))
        self.time = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.denoise = nn.Linear(3 * d, d)
        self.diffusion = Gaussian(torch.linspace(.0001, .02, self.settings["steps"]))

    def predict_clean(self, x, history, t):
        step = self.time(time_embedding(t, x.shape[-1]))
        return self.denoise(torch.cat((x, history, step), -1))

    def loss(self, sequence, lengths, target):
        h = self.history.encode(sequence, lengths)
        keep = torch.rand(len(target), 1, device=target.device) >= .1
        h = torch.where(keep, h, self.none)
        positive = self.history.items(target)
        if len(target) > 1:
            negative = (positive.sum(0, keepdim=True) - positive) / (len(target) - 1)
        else:
            neg_id = torch.randint(self.history.item_count - 1, target.shape, device=target.device)
            negative = self.history.items(neg_id + (neg_id >= target))
        t = torch.randint(len(self.diffusion.abar), target.shape, device=target.device)
        positive_prediction = self.predict_clean(self.diffusion.noise(positive, t), h, t)
        negative_prediction = self.predict_clean(self.diffusion.noise(negative, t), h, t)
        if self.settings.get("loss_type", "cosine") == "cosine":
            pos_loss = (1 - F.cosine_similarity(positive_prediction, positive, dim=-1)).square().mean()
            neg_loss = (1 - F.cosine_similarity(negative_prediction, negative, dim=-1)).square().mean()
        else:
            pos_loss, neg_loss = F.mse_loss(positive_prediction, positive), F.mse_loss(negative_prediction, negative)
        weight = self.settings["lambda"]
        return weight * pos_loss + (1 - weight) * F.softplus(self.settings["gamma"] * (pos_loss - neg_loss))

    def forward(self, sequence, lengths):
        h = self.history.encode(sequence, lengths)
        x = torch.randn_like(h)
        grid = torch.linspace(len(self.diffusion.abar) - 1, 0, self.settings["sampling_steps"]).long().tolist()
        for index, step in enumerate(grid):
            t = torch.full((len(x),), step, device=x.device, dtype=torch.long)
            w = self.settings["guidance"]
            clean = (1 + w) * self.predict_clean(x, h, t) - w * self.predict_clean(x, self.none.expand_as(h), t)
            a = self.diffusion.abar[step]
            previous = self.diffusion.abar[grid[index + 1]] if index + 1 < len(grid) else x.new_tensor(1.)
            epsilon = (x - a.sqrt() * clean) / (1 - a).sqrt()
            x = previous.sqrt() * clean + (1 - previous).sqrt() * epsilon
        return x @ self.history.items.weight[:self.history.item_count].T


class DiffRec(nn.Module):
    def __init__(self, n, config):
        super().__init__()
        self.n = n
        settings = config["baseline"]["diffrec"]
        d, self.time_width = settings["width"], 10
        self.time = nn.Linear(self.time_width, self.time_width)
        self.input = nn.Linear(n + self.time_width, d)
        self.output = nn.Linear(d, n)
        self.drop = nn.Dropout(.5)
        variance = settings["noise_scale"] * torch.linspace(settings["noise_min"], settings["noise_max"], settings["steps"], dtype=torch.float64)
        cumulative = 1 - variance
        beta = 1 - cumulative / F.pad(cumulative[:-1], (1, 0), value=1.)
        beta[0] = 1e-5
        self.diffusion = Gaussian(beta.float())
        self.reweight = settings["reweight"]
        self.sampling_steps = settings["sampling_steps"]

    def interaction_vector(self, sequence, target=None):
        if target is not None:
            sequence = torch.cat((sequence, target[:, None]), -1)
        # Padding never becomes an observed interaction.
        vector = torch.zeros(len(sequence), self.n + 1, device=sequence.device)
        return vector.scatter_(1, sequence, 1.)[:, :self.n]

    def denoise(self, x, t):
        step = self.time(time_embedding(t, self.time_width))
        return self.output(torch.tanh(self.input(torch.cat((self.drop(x), step), -1))))

    def loss(self, sequence, lengths, target):
        clean = self.interaction_vector(sequence, target)
        t = torch.randint(len(self.diffusion.abar), target.shape, device=target.device)
        prediction = self.denoise(self.diffusion.noise(clean, t), t)
        mse = (prediction - clean).square().mean(-1)
        if self.reweight:
            abar = self.diffusion.abar
            snr = abar / (1 - abar)
            weights = (snr[(t - 1).clamp_min(0)] - snr[t]).masked_fill(t == 0, 1.)
            mse = mse * weights
        return mse.mean()

    def forward(self, sequence, lengths):
        x = self.interaction_vector(sequence)
        if self.sampling_steps:
            t = torch.full((len(x),), self.sampling_steps - 1, device=x.device, dtype=torch.long)
            x = self.diffusion.noise(x, t)
        for step in reversed(range(len(self.diffusion.abar))):
            t = torch.full((len(x),), step, device=x.device, dtype=torch.long)
            clean = self.denoise(x, t)
            x = self.diffusion.posterior1[step] * clean + self.diffusion.posterior2[step] * x
        return x


class DDSR(nn.Module):
    def __init__(self, n, config, artifact):
        super().__init__()
        self.n = n
        self.m, self.k = artifact["codebooks"], artifact["codebook_size"]
        self.settings = config["baseline"]["ddsr"]
        self.register_buffer("codes", artifact["codes"])
        self.register_buffer("centroids", artifact["centroids"])
        d, length = config["hidden_size"], config["history_length"]
        self.items = nn.Embedding(self.m * (self.k + 1), d)
        self.positions = nn.Embedding(length * self.m, d)
        self.time = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, 2, 4 * d, config["baseline"]["dropout"],
                    activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, self.k)
        # Uniform transition: diagonal beta, off-diagonal (1-beta)/(K-1).
        steps = self.settings["steps"]
        times = torch.arange(steps + 1, dtype=torch.float64) / steps
        retention = torch.cos((times + .008) / 1.008 * math.pi / 2).square()
        retention = retention / retention[0]
        factors = (retention[1:] / retention[:-1]).clamp(.001, 1.)
        self.register_buffer("cumulative", torch.cat((torch.ones(1), factors.cumprod(0).float())))
        self.register_buffer("causal", (torch.arange(length * self.m) // self.m)[None] >
                             (torch.arange(length * self.m) // self.m)[:, None])

    def predict_codes(self, current, valid, t):
        batch, length, _ = current.shape
        offsets = torch.arange(self.m, device=current.device) * (self.k + 1)
        hidden = self.items((current + offsets).flatten(1))
        hidden = hidden + self.positions(torch.arange(length * self.m, device=current.device))
        hidden = hidden + self.time(time_embedding(t, hidden.shape[-1]))[:, None]
        # Key mask plus causal mask must not leave padded queries all masked.
        # Padding is zeroed and allowed as a zero prefix to keep attention finite.
        hidden = hidden * valid.repeat_interleave(self.m, -1)[..., None]
        hidden = self.norm(self.encoder(hidden, mask=self.causal))
        return self.head(hidden).reshape(batch, length, self.m, self.k)

    def corrupt(self, codes, valid, t):
        keep = torch.rand_like(codes.float()) < self.cumulative[t, None, None]
        return torch.where(valid[..., None] & ~keep, torch.randint_like(codes, self.k), codes)

    def loss(self, sequence, lengths, target):
        valid = sequence < self.n
        codes = self.codes[sequence]
        t = torch.randint(len(self.cumulative), target.shape, device=target.device)
        noisy = self.corrupt(codes, valid, t)
        # Every historical position predicts the following clean item's codes.
        next_items = torch.cat((sequence[:, 1:], target[:, None]), -1)
        labels = self.codes[next_items]
        logits = self.predict_codes(noisy, valid, t)
        return F.cross_entropy(logits[valid].reshape(-1, self.k), labels[valid].reshape(-1))

    def reverse_probabilities(self, predicted, current, step, previous):
        # Eq. 7: sum_x0 q(x_prev,x_t|x0) p_theta(x0|x_t), normalized.
        a = self.cumulative[previous]
        ratio = self.cumulative[step] / a.clamp_min(1e-30)
        prior = a * predicted + (1 - a) / self.k
        likelihood = torch.full_like(prior, (1 - ratio) / self.k)
        likelihood.scatter_add_(-1, current.clamp_max(self.k - 1)[..., None], torch.full_like(current[..., None], float(ratio), dtype=prior.dtype))
        posterior = prior * likelihood
        return posterior / posterior.sum(-1, keepdim=True)

    def forward(self, sequence, lengths):
        valid = sequence < self.n
        current = self.codes[sequence].clone()  # History start, never pure noise.
        steps = self.settings["steps"]
        skip = self.settings["skip"]
        grid = list(range(steps, 0, -skip)) + [0]
        for step, previous in zip(grid, grid[1:]):
            t = torch.full((len(sequence),), step, device=sequence.device, dtype=torch.long)
            predicted = self.predict_codes(current, valid, t).softmax(-1)
            posterior = self.reverse_probabilities(predicted, current, step, previous)
            sampled = torch.multinomial(posterior.flatten(0, -2), 1).reshape_as(current)
            current = torch.where(valid[..., None], sampled, current)
        # Full-catalog nearest-codeword decoding, including semantic collisions.
        selected = current[:, -1]
        scores = torch.zeros(len(sequence), self.n, device=sequence.device)
        for level in range(self.m):
            query = self.centroids[level][selected[:, level]]
            distances = torch.cdist(query, self.centroids[level]).square()
            scores -= distances[:, self.codes[:self.n, level]]
        return scores


def make_baseline(output, config, device):
    n, method = metadata(output)["item_count"], config["method"]
    constructors = {"caser": Caser, "gru4rec": GRU4Rec, "bert4rec": BERT4Rec, "diffrec": DiffRec}
    if method in {"preferdiff", "ddsr"}:
        artifact = torch.load(config["semantic_file"], weights_only=True, map_location="cpu")
        model = (PreferDiff if method == "preferdiff" else DDSR)(n, config, artifact)
    else:
        model = constructors[method](n, config)
    return model.to(device)


@torch.no_grad()
def evaluate_baseline(model, values, config, device, split, streams):
    model.eval()
    results = []
    for stream in range(streams):
        totals = {}
        with evaluation_rng(config["seed"], split, stream, device):
            for sequence, lengths, target in loader(values, config["training"]["eval_batch_size"]):
                scores = model(sequence.to(device), lengths.to(device))
                if not torch.isfinite(scores).all() or scores.shape[1] != model.n:
                    raise FloatingPointError("Invalid baseline full-catalog scores")
                for key, value in ranking_totals(scores, target.to(device)).items():
                    totals[key] = totals.get(key, 0.) + value
        results.append({key: value / len(values["target"]) for key, value in totals.items()})
    return {**{key: sum(row[key] for row in results) / streams for key in results[0]},
            "examples": len(values["target"]), "streams": streams, "per_stream": results}


def train_baseline(output, config, device):
    output = Path(output)
    seed_all(config["seed"])
    settings = config["training"]
    model = make_baseline(output, config, device)
    optimizer = make_baseline_optimizer(model, config)
    training, validation = load_split(output, "train"), load_split(output, "val")
    batches = loader(training, settings["batch_size"], shuffle=True, seed=config["seed"])
    best, best_epoch, best_seconds, best_state = None, 0, 0., None
    started = time.perf_counter()
    step = 0
    for epoch in training_epochs(settings["epochs"]):
        model.train()
        total, examples = 0., 0
        for sequence, lengths, target in batches:
            loss = model.loss(sequence.to(device), lengths.to(device), target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {config['method']} loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if settings["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            for group in optimizer.param_groups:
                group["lr"] = baseline_learning_rate(settings, step)
            optimizer.step()
            step += 1
            total += float(loss.detach()) * len(target)
            examples += len(target)
        metrics = None
        if epoch % settings["validation_interval"] == 0 or epoch == settings["epochs"]:
            metrics = evaluate_baseline(model, validation, config, device, "val", settings["validation_streams"])
            if better(metrics, best):
                best, best_epoch, best_seconds = metrics, epoch, time.perf_counter() - started
                # Keep the validation winner on CPU for the immediate test, without a model file.
                best_state = {"model": {key: value.detach().cpu().clone()
                                        for key, value in model.state_dict().items()},
                              "epoch": epoch, "validation": metrics}
        append_log(output / "baseline_log.jsonl", {"stage": config["method"], "epoch": epoch,
                   "loss": total / examples, "validation": metrics, "best_epoch": best_epoch})
        if metrics is not None and epoch - best_epoch >= settings["patience"]:
            break
    write_json(output / "training.json", {"epochs_completed": epoch, "best_epoch": best_epoch,
               "best_validation": best, "selection_split": "val", "objective": config["method"],
               "epoch_limit": settings["epochs"] or None, "selected_checkpoint_seconds": best_seconds,
               "elapsed_seconds": time.perf_counter() - started})
    return best_state


def test_baseline(output, config, device, checkpoint=None):
    output = Path(output)
    if config["method"] == "sasrec":
        model = SASRec(metadata(output)["item_count"], config).to(device)
        checkpoint = torch.load(Path(config["candidate_file"]).parent / "sasrec_best.pt", weights_only=True, map_location=device)
        model.load_state_dict(checkpoint["model"])
        metrics = evaluate_proposer(model, load_split(output, "test"), config["training"]["eval_batch_size"], device)
        result = {**metrics, "examples": len(load_split(output, "test")["target"]), "streams": 1, "per_stream": [metrics]}
    else:
        model = make_baseline(output, config, device)
        if checkpoint is None:
            # Compatibility with fits already running when the retention policy changed.
            checkpoint = torch.load(output / "baseline_best.pt", weights_only=True, map_location=device)
        model.load_state_dict(checkpoint["model"])
        result = evaluate_baseline(model, load_split(output, "test"), config, device, "test", config["training"]["test_streams"])
    report = {"dataset": config["dataset"], "method": config["method"], "seed": config["seed"],
              "smoke": config["smoke"], "best_epoch": checkpoint["epoch"], "selection_split": "val", "test": result}
    write_json(output / "report.json", report)
    print(f"[test] {report}", flush=True)
    return report
