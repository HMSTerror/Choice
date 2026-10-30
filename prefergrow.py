"""Rank-one PreferGrow corruption, adapted to the common full-catalog evaluator.

Equations follow Hugo-Chinn/PreferGrow, graph_lib.py (HybridWise/AdaptiveWise),
revision 7451996bbd6d289cf96457870cdc45a79fab8311. The common denoiser,
validation selection and tau-leap evaluator are shared with ChoiceGrow.
"""
import torch
from torch import nn
import torch.nn.functional as F

from diffusion import ChoiceProcess, Context


class PreferGrowProcess(ChoiceProcess):
    def __init__(self, artifact, config):
        nn.Module.__init__(self)
        self.settings = config["diffusion"]
        self.destination = config["destination"]
        self.catalog_count = len(artifact["base"])
        self.item_count = self.catalog_count + 1
        self.variant = config.get("prefergrow", {}).get("variant", "hybrid")
        gamma = config.get("prefergrow", {}).get("gamma", .9999)
        distribution = torch.full((self.item_count,), gamma / self.catalog_count)
        distribution[-1] = 1 - gamma
        self.register_buffer("base", distribution)

    def context(self, sequence, candidates, model=None):
        rho = (model.global_destination.softmax(-1) if self.variant == "adaptive" else self.base)
        return Context(rho.expand(len(sequence), -1), ())

    def forward_probability(self, source, sigma, context):
        retention = (-sigma[:, None]).exp()
        return retention * F.one_hot(source, self.item_count) + (1 - retention) * context.rho

    def terminal_reference(self, sigma, context):
        return context.rho

    def incoming_rates(self, current, context):
        return context.rho.gather(1, current[:, None]).expand_as(context.rho).scatter(1, current[:, None], 0.)
