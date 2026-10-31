"""History-fixed CTMC, exact seven-coordinate forward kernel, and reverse sampler."""
# Note: .agents/notes/implemented/architecture/2026-10-29-minimal-choice.md
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


def sample_training_noise(batch_size, settings, device):
    """NewChoice losses.py sigma-derivative proposal, corrected to uniform-grid expectation."""
    grid = torch.linspace(1., .001, settings["training_intervals"] + 1, device=device)
    log_range = math.log(settings["sigma_max"] / settings["sigma_min"])
    sigma = settings["sigma_min"] * torch.exp(grid * log_range)
    derivative = sigma * log_range
    mode = settings.get("time_sampling", "uniform")
    if mode == "uniform":
        indices = torch.randint(len(grid), (batch_size,), device=device)
        correction = 1.
    elif mode == "sigma_importance":
        probability = .1 / len(grid) + .9 * derivative.detach() / derivative.detach().sum()
        indices = torch.multinomial(probability, batch_size, replacement=True)
        correction = (1. / (len(grid) * probability[indices])).detach()
    else:
        raise ValueError(f"Unsupported time_sampling: {mode}")
    return sigma[indices], derivative[indices] * correction


@dataclass
class Context:
    rho: torch.Tensor
    block_mass: tuple


def random_partition(item_count, fine_size, coarse_size, seed):
    if coarse_size % fine_size:
        raise ValueError("coarse_size must be a multiple of fine_size for nested blocks")
    order = torch.randperm(item_count, generator=torch.Generator().manual_seed(seed))
    rank = torch.empty_like(order)
    rank[order] = torch.arange(item_count)
    return torch.stack((rank // fine_size, rank // coarse_size, torch.zeros_like(rank)))


class ChoiceProcess(nn.Module):
    def __init__(self, artifact, config):
        super().__init__()
        self.settings = config["diffusion"]
        self.destination = config["destination"]
        if self.destination.get("candidate_gate", "fixed_mass") not in {"fixed_mass", "dynamic_mass"}:
            raise ValueError("Unknown candidate gate")
        for name in ("base", "blocks", "neighbor_ids", "neighbor_weights"):
            self.register_buffer(name, artifact[name])
        self.item_count = len(self.base)
        alpha = self.settings["reset_rate"]
        scale = self.settings.get("rate_scale", 1.)
        self.rates = tuple(scale * (1 - alpha) * value for value in self.settings["level_rates"])
        self.reset_rate = scale * alpha
        # Column-source convention: bridge P_l G represents the paper's G P_l.
        # P_root G = P_root, so the three-level representation needs seven coordinates.
        self.basis = (("I", -1), ("P", 0), ("P", 1), ("P", 2),
                      ("G", -1), ("B", 0), ("B", 1))

        def product(left, right):
            a, i = left
            b, j = right
            if a == "I":
                result = right
            elif b == "I":
                result = left
            elif a == "P" and b == "P":
                result = ("P", max(i, j))
            elif a == "G":
                result = ("G", -1)
            elif a == "P" and b == "G":
                result = ("B", i)
            elif a == "P" and b == "B":
                result = ("B", max(i, j))
            else:  # B_i P_j = B_i, B_i G = B_i, B_i B_j = B_i.
                result = left
            return ("P", 2) if result == ("B", 2) else result

        table = torch.tensor([[self.basis.index(product(a, b)) for b in self.basis]
                              for a in self.basis])
        structure = F.one_hot(table, len(self.basis)).permute(0, 2, 1).double()
        generator = torch.tensor([-sum(self.rates) - self.reset_rate, *self.rates, self.reset_rate, 0., 0.],
                                 dtype=torch.float64)
        self.register_buffer("algebra", torch.einsum("l,ldr->dr", generator, structure))

    def context(self, sequence, candidates, model=None):
        base = self.base.expand(len(sequence), -1)
        if self.destination.get("mode", "user") == "shared":
            return Context(base, tuple(self.block_sum(base, level) for level in range(3)))
        mask = torch.zeros_like(base).scatter(1, candidates, 1.0)
        weights_base = torch.ones_like(base) if self.destination.get("weighting") == "uniform" else base
        inside, outside = weights_base * mask, weights_base * (1 - mask)
        if torch.any(inside.sum(-1) <= 0) or torch.any(outside.sum(-1) <= 0):
            raise ValueError("The candidate set must be a nonempty proper catalog subset")
        mass = (model.candidate_mass(sequence) if self.destination.get("candidate_gate") == "dynamic_mass"
                else self.destination["candidate_mass"])
        rho = mass * inside / inside.sum(-1, keepdim=True)
        rho = rho + (1 - mass) * outside / outside.sum(-1, keepdim=True)
        valid = sequence < self.item_count
        safe = sequence.clamp_max(self.item_count - 1)
        age = sequence.shape[1] - 1 - torch.arange(sequence.shape[1], device=sequence.device)
        weights = valid * torch.exp(-self.destination["history_decay"] * age)
        ids = self.neighbor_ids[safe]
        neighbor_weights = self.neighbor_weights[safe] * weights[..., None]
        recent = torch.zeros_like(base).scatter_add(1, ids.flatten(1), neighbor_weights.flatten(1))
        recent = recent / weights.sum(-1, keepdim=True)
        mix = self.destination["neighbor_mix"]
        rho = (1 - mix) * rho + mix * recent
        rho = rho / rho.sum(-1, keepdim=True)
        block_mass = tuple(self.block_sum(rho, level) for level in range(3))
        return Context(rho, block_mass)

    def block_sum(self, values, level):
        ids = self.blocks[level].expand(len(values), -1)
        totals = torch.zeros_like(values).scatter_add(1, ids, values)
        return totals.gather(1, ids)

    def projection(self, source, level, context):
        ids = self.blocks[level]
        mask = ids[None] == ids[source, None]
        denominator = context.block_mass[level].gather(1, source[:, None])
        return context.rho * mask / denominator

    def bridge(self, level, context):
        return context.rho * self.block_sum(self.base.expand_as(context.rho), level) / context.block_mass[level]

    def coefficients(self, sigma):
        unique, inverse = torch.unique(sigma.double().reshape(-1), return_inverse=True)
        values = torch.matrix_exp(unique[:, None, None] * self.algebra)[..., 0]
        return values[inverse].to(self.base.dtype)

    def forward_probability(self, source, sigma, context):
        weights = self.coefficients(sigma)
        result = weights[:, :1] * F.one_hot(source, self.item_count)
        for level in range(3):
            result = result + weights[:, level + 1:level + 2] * self.projection(source, level, context)
        result = result + weights[:, 4:5] * self.base
        for level in range(2):
            result = result + weights[:, level + 5:level + 6] * self.bridge(level, context)
        if not torch.isfinite(result).all() or torch.any(result <= 0):
            raise FloatingPointError("Invalid forward probability")
        return result

    def terminal_reference(self, sigma, context):
        sigma = torch.full((len(context.rho),), float(sigma), device=self.base.device,
                           dtype=self.base.dtype)
        weights = self.coefficients(sigma)
        result = (weights[:, :1] + weights[:, 4:5]) * self.base
        for level in range(3):
            coefficient = weights[:, level + 1:level + 2]
            if level < 2:
                coefficient = coefficient + weights[:, level + 5:level + 6]
            result = result + coefficient * self.bridge(level, context)
        return result / result.sum(-1, keepdim=True)

    def incoming_rates(self, current, context):
        """Return Q_row[y, current], with the self edge removed."""
        rate = self.reset_rate * self.base[current, None].expand_as(context.rho)
        destination = context.rho.gather(1, current[:, None])
        for level, intensity in enumerate(self.rates):
            ids = self.blocks[level]
            members = ids[None] == ids[current, None]
            denominator = context.block_mass[level].gather(1, current[:, None])
            rate = rate + intensity * destination * members / denominator
        return rate.scatter(1, current[:, None], 0.0)

    def entropy(self, log_score, probability, current, context):
        reference = probability.log() - probability.gather(1, current[:, None]).log()
        delta = log_score - reference
        return (self.incoming_rates(current, context) * reference.exp()
                * (torch.expm1(delta) - delta)).sum(-1)

    def loss(self, model, sequence, target, candidates):
        settings = self.settings
        sigma, entropy_weight = sample_training_noise(len(target), settings, target.device)
        context = self.context(sequence, candidates, model)
        probability = self.forward_probability(target, sigma, context)
        current = torch.multinomial(probability, 1).squeeze(1)
        score = model(sequence, current, sigma)
        # The sole training objective: derivative-weighted denoising score entropy.
        values = entropy_weight * self.entropy(score, probability, current, context)
        if self.destination.get("candidate_gate") == "dynamic_mass" or hasattr(model, "global_destination"):
            # Include the discrete-sampling score-function term when learning Q.
            logp = probability.gather(1, current[:, None]).squeeze(1).log()
            values = values + values.detach() * (logp - logp.detach())
        weight = self.settings.get("guided_negative_weight", 0.)
        if weight:
            target_score = score.gather(1, target[:, None])
            negative_score = score.gather(1, candidates)
            valid = candidates != target[:, None]
            ranking = (F.softplus(negative_score - target_score) * valid).sum(-1) / valid.sum(-1).clamp_min(1)
            values = values + weight * ranking
        return values.mean()

    @torch.no_grad()
    def sample_scores(self, model, sequence, candidates, guidance):
        settings = self.settings
        context = self.context(sequence, candidates, model)
        history = model.encode_history(sequence)
        initial = (self.base.expand_as(context.rho) if settings.get("initialization") == "shared_base"
                   else self.terminal_reference(settings["sigma_max"], context))
        if hasattr(self, "reference_process"):
            reference_context = self.reference_process.context(sequence, candidates, model)
            initial = self.reference_process.terminal_reference(settings["sigma_max"], reference_context)
        current = torch.multinomial(initial, 1).squeeze(1)
        grid = torch.linspace(settings["sigma_max"], settings["sigma_min"],
                              settings["reverse_intervals"] + 1, device=sequence.device)
        step_count = torch.zeros(len(sequence), device=sequence.device)
        cap_count = 0
        for high, low in zip(grid[:-1], grid[1:]):
            remaining = torch.full((len(sequence),), float(high - low), device=sequence.device)
            for _ in range(settings["max_substeps"]):
                active = remaining > 1e-7
                if not active.any():
                    break
                sigma = low + remaining
                scores = model.guided_scores(history, current, sigma, guidance)
                incoming = self.incoming_rates(current, context)
                log_rates = incoming.log() + scores
                log_exit = torch.logsumexp(log_rates, -1)
                if not torch.isfinite(log_exit).all():
                    raise FloatingPointError("Nonfinite reverse exit rate")
                effective = log_exit.clamp_max(settings["log_exit_cap"])
                cap_count += int(((effective < log_exit) & active).sum())
                dt = torch.minimum(remaining, settings["reverse_cfl"] * torch.exp(-effective))
                if torch.any(active & (dt <= 0)):
                    raise FloatingPointError("Reverse step underflow")
                event = (torch.rand_like(remaining) < dt * effective.exp()) & active
                destination = torch.multinomial((log_rates - log_exit[:, None]).exp(), 1).squeeze(1)
                current = torch.where(event, destination, current)
                remaining = (remaining - dt).clamp_min(0)
                step_count += active
            if torch.any(remaining > 1e-7):
                raise RuntimeError("Reverse substep budget exceeded")
        endpoint = torch.full((len(sequence),), settings["sigma_min"], device=sequence.device)
        return model.guided_scores(history, current, endpoint, guidance), {
            "mean_reverse_substeps": float(step_count.mean()), "exit_cap_activations": cap_count}
