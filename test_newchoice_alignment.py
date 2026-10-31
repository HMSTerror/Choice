"""Training protocol regressions for the retained NewChoice denoiser."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from diffusion import ChoiceProcess
from experiments import arm_config, config_for
from models import Denoiser


ROOT = Path(__file__).resolve().parent


class NewChoiceAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config.json").read_text())
        self.config.update(dataset="Beauty", hidden_size=8, history_length=4)
        self.config["training"].update(heads=2, dropout=.25)
        torch.set_num_threads(2)

    def test_choice_defaults_match_requested_newchoice_protocol(self):
        self.assertEqual(self.config["training"]["patience"], 100)
        self.assertEqual(self.config["diffusion"]["time_sampling"], "sigma_importance")
        self.assertEqual(self.config["training"]["denoiser"], "newchoice_ddit")
        suite = json.loads((ROOT / "paper_suite.json").read_text())
        caser = arm_config(config_for(self.config, "Beauty", 100), "caser", suite)
        self.assertEqual(caser["training"]["patience"], 50)
        prefer = arm_config(config_for(self.config, "Beauty", 100), "prefergrow", suite)
        self.assertEqual(prefer["diffusion"]["time_sampling"], "uniform")

    def test_masked_history_and_bounded_guidance_keep_padding_out_of_scores(self):
        model = Denoiser(12, self.config).eval()
        self.assertEqual(type(model.history_blocks[0]).__name__, "DDiTBlock_UCUR")
        self.assertEqual(type(model.output_layer).__name__, "DDitFinalLayer4Rec")
        sequence = torch.tensor([[12, 12, 1, 2], [12, 3, 4, 5], [12, 12, 12, 12]])
        current = torch.tensor([4, 7, 2]);sigma = torch.tensor([.001, .3, 10.])
        history = model.encode_history(sequence)
        torch.testing.assert_close(history[-1], torch.zeros(8))
        with torch.no_grad():
            model.items.weight[12].fill_(100)
        torch.testing.assert_close(history, model.encode_history(sequence))
        for scores in (model(sequence, current, sigma), model.guided_scores(history, current, sigma, 2.)):
            self.assertTrue(torch.isfinite(scores).all())
            self.assertLessEqual(float(scores.abs().max()), self.config["training"]["log_score_cap"])
            torch.testing.assert_close(scores.gather(1, current[:, None]), torch.zeros(3, 1))

    def test_importance_correction_preserves_uniform_grid_objective_and_gradient(self):
        from diffusion import sample_training_noise
        settings = self.config["diffusion"]
        grid = torch.linspace(1., .001, settings["training_intervals"] + 1)
        ratio = torch.log(torch.tensor(settings["sigma_max"] / settings["sigma_min"]))
        derivative = settings["sigma_min"] * torch.exp(grid * ratio) * ratio
        proposal = .1 / len(grid) + .9 * derivative / derivative.sum()
        indices = torch.arange(len(grid))
        with patch("diffusion.torch.multinomial", return_value=indices) as draw:
            sigma, weight = sample_training_noise(len(grid), settings, torch.device("cpu"))
        torch.testing.assert_close(draw.call_args.args[0], proposal)
        self.assertTrue(draw.call_args.kwargs["replacement"])
        torch.testing.assert_close(sigma, derivative / ratio)
        parameter = torch.tensor(.7, requires_grad=True)
        entropy = (sigma + parameter).square()
        weighted = (proposal * weight * entropy).sum()
        expected = (derivative * entropy).mean()
        torch.testing.assert_close(weighted, expected)
        torch.testing.assert_close(torch.autograd.grad(weighted, parameter, retain_graph=True)[0],
                                   torch.autograd.grad(expected, parameter)[0])
        uniform = deepcopy(settings);uniform["time_sampling"] = "uniform"
        with patch("diffusion.torch.randint", return_value=indices):
            _, uniform_weight = sample_training_noise(len(grid), uniform, torch.device("cpu"))
        torch.testing.assert_close(uniform_weight, derivative)


if __name__ == "__main__":
    unittest.main()
