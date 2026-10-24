"""Keep FP32 Adam moments in host RAM, staging one parameter's state on CUDA.

Parameters, gradients and update arithmetic stay on CUDA. The policy's existing
per-component clipping and nonfinite-step guard remain authoritative. No optimizer
state is quantized, and saved checkpoints keep PyTorch's AdamW state schema.
"""
from __future__ import annotations

import json

import torch
from torch.optim import _functional

from lerobot.policies.molmoact2 import configuration_molmoact2 as upstream


class CPUStateMolmoAdamW(upstream.MolmoAct2AdamW):
    """FP32/CUDA-only AdamW with CPU-resident first and second moments."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for group in self.param_groups:
            if group.get("amsgrad") or group.get("capturable") or group.get("differentiable"):
                raise ValueError("CPU-state Molmo AdamW supports the workshop AdamW preset only")
            if any(p.dtype != torch.float32 or p.device.type != "cuda" for p in group["params"]):
                raise ValueError("CPU-state Molmo AdamW requires FP32 CUDA trainable parameters")

    @staticmethod
    def _host_copy(tensor):
        result = torch.empty_like(tensor, device="cpu", pin_memory=True)
        result.copy_(tensor)
        return result

    @torch.no_grad()
    def _step_native_non_bfloat16(self):
        devices = set()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                devices.add(parameter.device)
                state = self.state[parameter]
                if not state:
                    state["step"] = torch.zeros((), dtype=torch.float32, device="cpu")
                    state["exp_avg"] = torch.zeros_like(parameter, device="cpu", pin_memory=True)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, device="cpu", pin_memory=True)
                # These transfers are ordered on the same stream as the update
                # and copy-back. Freed CUDA staging storage can be reused by the
                # next parameter without retaining model-sized state on device.
                first = state["exp_avg"].to(parameter.device, non_blocking=True)
                second = state["exp_avg_sq"].to(parameter.device, non_blocking=True)
                _functional.adamw(
                    [parameter], [parameter.grad], [first], [second], [], [state["step"]],
                    foreach=True, capturable=False, differentiable=False, fused=False,
                    grad_scale=None, found_inf=None, has_complex=False,
                    amsgrad=False, beta1=group["betas"][0], beta2=group["betas"][1],
                    lr=group["lr"], weight_decay=group["weight_decay"],
                    eps=group["eps"], maximize=group["maximize"],
                )
                state["exp_avg"].copy_(first, non_blocking=True)
                state["exp_avg_sq"].copy_(second, non_blocking=True)
                del first, second
        # Complete async host writes before a subsequent update or checkpoint
        # reads the moments. This also makes CPU serialization deterministic.
        for device in devices:
            torch.cuda.synchronize(device)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for state in self.state.values():
            if "step" in state:
                state["step"] = state["step"].cpu()
            for name in ("exp_avg", "exp_avg_sq"):
                if name in state:
                    state[name] = self._host_copy(state[name])
        moment_bytes = sum(
            value.numel() * value.element_size()
            for state in self.state.values() for name, value in state.items()
            if name in ("exp_avg", "exp_avg_sq")
        )
        print("WORKSHOP_OPTIMIZER_OFFLOAD " + json.dumps({
            "moment_storage": "cpu", "moment_dtype": "float32",
            "moment_gib": moment_bytes / 2**30, "quantized": False,
            "parameter_states": len(self.state),
        }), flush=True)


def install_cpu_optimizer_state():
    """Change only the optimizer implementation returned by the policy preset."""
    upstream.MolmoAct2AdamW = CPUStateMolmoAdamW
