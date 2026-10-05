"""CPU regression tests for the DiT continuation's freeze and loading boundaries."""

import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_dit_only", REPO / "scripts/train_dit_only.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.backbone = nn.Module()
        self.backbone.llm = nn.Linear(2, 2, device="cpu")
        self.backbone.visual = nn.Linear(2, 2, device="cpu")
        self.backbone.projector = nn.Linear(2, 2, device="cpu")
        self.action_head = nn.Module()
        self.action_head.config = SimpleNamespace()
        self.action_head.model = nn.Module()
        self.action_head.model.transformer_blocks = nn.ModuleList(
            [nn.Linear(2, 2, device="cpu") for _ in range(4)]
        )
        self.action_head.model.timestep_encoder = nn.Linear(2, 2, device="cpu")
        self.action_head.model.proj_out_2 = nn.Linear(2, 2, device="cpu")
        for name in (
            "state_encoder",
            "action_encoder",
            "action_decoder",
            "vlln",
            "vl_self_attention",
            "position_embedding",
        ):
            setattr(self.action_head, name, nn.Linear(2, 2, device="cpu"))

    def forward(self, x):
        x = self.action_head.action_encoder(self.backbone.llm(x))
        for block in self.action_head.model.transformer_blocks:
            x = torch.tanh(block(x))
        return self.action_head.action_decoder(x)


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()

    def freeze(self, last_n=0):
        with redirect_stdout(io.StringIO()):
            return launcher.freeze_dit_only(self.model, last_n)

    def test_only_the_whole_dit_is_trainable(self):
        allowed = self.freeze()
        expected = {
            name
            for name, _ in self.model.named_parameters()
            if name.startswith("action_head.model.")
        }
        self.assertEqual(allowed, expected)
        self.assertEqual(launcher.assert_dit_only(self.model, allowed), 36)
        self.assertFalse(self.model.action_head.tune_projector)
        self.assertFalse(self.model.action_head.tune_vlln)
        self.assertTrue(self.model.action_head.tune_diffusion_model)

    def test_only_last_n_blocks_are_trainable(self):
        allowed = self.freeze(2)
        expected = {
            f"action_head.model.transformer_blocks.{i}.{p}"
            for i in (2, 3)
            for p in ("weight", "bias")
        }
        self.assertEqual(allowed, expected)
        self.assertEqual(launcher.assert_dit_only(self.model, allowed), 12)

    def test_non_dit_leak_aborts(self):
        self.freeze()
        for name in (
            "backbone.llm",
            "backbone.visual",
            "backbone.projector",
            "action_head.action_decoder",
            "action_head.vl_self_attention",
        ):
            with self.subTest(name=name):
                parameter = self.model.get_submodule(name).weight
                parameter.requires_grad_(True)
                with self.assertRaisesRegex(RuntimeError, "Invalid DiT-only"):
                    launcher.assert_dit_only(self.model)
                parameter.requires_grad_(False)

    def test_block_selection_leak_aborts_even_inside_dit(self):
        allowed = self.freeze(1)
        self.model.action_head.model.transformer_blocks[0].weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "Invalid DiT-only"):
            launcher.assert_dit_only(self.model, allowed)

    def test_invalid_block_count_aborts(self):
        for count in (-1, 5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                launcher.freeze_dit_only(self.model, count)

    def test_no_trainable_parameters_aborts(self):
        self.model.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, "No DiT"):
            launcher.assert_dit_only(self.model)

    def test_backward_through_frozen_decoder_updates_only_selected_dit(self):
        allowed = self.freeze(2)
        previous = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=0.01
        )
        self.model(torch.ones(3, 2, device="cpu")).square().mean().backward()
        optimizer.step()
        changed = {
            name for name, p in self.model.named_parameters() if not torch.equal(previous[name], p)
        }
        self.assertEqual(changed, allowed)
        self.assertTrue(
            all(p.grad is None for name, p in self.model.named_parameters() if name not in allowed)
        )

    def test_checkpointed_blocks_preserve_gradients(self):
        self.freeze(2)
        reference = FakeModel()
        reference.load_state_dict(self.model.state_dict())
        with redirect_stdout(io.StringIO()):
            launcher.freeze_dit_only(reference, 2)
            launcher.checkpoint_blocks(self.model.action_head.model)
        x = torch.ones(3, 2, device="cpu")
        self.model(x).sum().backward()
        reference(x).sum().backward()
        for (name, p), (_, expected) in zip(
            self.model.named_parameters(), reference.named_parameters()
        ):
            if p.requires_grad:
                self.assertTrue(torch.allclose(p.grad, expected.grad), name)

    def test_parameter_table_counts_cover_every_parameter(self):
        self.freeze(1)
        with redirect_stdout(io.StringIO()):
            rows = launcher.parameter_table(self.model)
        self.assertEqual(
            sum(row["parameters"] for row in rows), sum(p.numel() for p in self.model.parameters())
        )
        self.assertEqual(sum(row["parameters"] for row in rows if row["trainable"]), 6)


class WeightLoadingTests(unittest.TestCase):
    def test_complete_loading_passes(self):
        with redirect_stdout(io.StringIO()):
            launcher.assert_complete_loading({"missing_keys": [], "unexpected_keys": []})

    def test_any_missing_key_aborts_including_optional_and_frozen_weights(self):
        for name in (
            "backbone.llm.weight",
            "action_head.mask_token",
            "action_head.model.transformer_blocks.0.weight",
        ):
            with self.subTest(name=name), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "Incomplete checkpoint"):
                    launcher.assert_complete_loading({"missing_keys": [name]})

    def test_unexpected_mismatched_or_loader_error_aborts(self):
        for name in ("unexpected_keys", "mismatched_keys", "error_msgs"):
            with (
                self.subTest(name=name),
                redirect_stdout(io.StringIO()),
                self.assertRaises(RuntimeError),
            ):
                launcher.assert_complete_loading({name: ["bad tensor"]})


class SavedConfigTests(unittest.TestCase):
    def test_complete_saved_data_and_preprocessing_survive_training_overrides(self):
        import sys

        import yaml

        groot = Path.home() / "Isaac-GR00T-N1.7"
        if (
            not groot.exists()
            or not (Path(launcher.DEFAULT_INIT) / "experiment_cfg/config.yaml").is_file()
        ):
            self.skipTest("Pinned GR00T checkout or saved init config unavailable")
        sys.path.insert(0, str(groot))
        from gr00t.configs.base_config import _build_safe_tree

        saved = yaml.safe_load(
            (Path(launcher.DEFAULT_INIT) / "experiment_cfg/config.yaml").read_text()
        )
        args = SimpleNamespace(
            load_ckpt=Path(launcher.DEFAULT_INIT),
            output_dir=Path("/tmp/fake-dit-output"),
            smoke=50,
            steps=20000,
            batch=8,
            grad_accum=4,
            lr=2e-5,
            schedule="cosine",
            warmup=1000,
            save_steps=5000,
            save_total_limit=4,
            resume=False,
            optim="adamw",
            grad_ckpt=False,
        )
        original = yaml.safe_dump(saved)
        config = launcher.resolved_config(args, saved)
        self.assertEqual(_build_safe_tree(asdict(config.data)), saved["data"])
        model = config.model.to_filtered_dict(exclude_augment=False)
        for name, value in saved["model"].items():
            if name not in (
                "tune_llm",
                "tune_visual",
                "tune_projector",
                "tune_vlln",
                "tune_diffusion_model",
            ):
                self.assertEqual(model.get(name), value, name)
        self.assertEqual(yaml.safe_dump(saved), original)
        self.assertEqual(config.training.max_steps, 50)
        self.assertEqual(config.training.gradient_accumulation_steps, 4)
        self.assertFalse(config.training.skip_weight_loading)


if __name__ == "__main__":
    unittest.main()
