"""Check CUDA AdamW update/state parity, nonfinite guard and checkpoint recovery."""
import copy
import io
import json

import torch

from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2AdamW
from molmo_optimizer_offload import CPUStateMolmoAdamW


torch.manual_seed(771)
reference_parameters = [torch.nn.Parameter(torch.randn(*shape, device="cuda"))
                        for shape in ((7, 11), (4096, 8), (17,))]
offloaded_parameters = [torch.nn.Parameter(p.detach().clone()) for p in reference_parameters]


def make(optimizer_class, parameters):
    return optimizer_class([
        {"params": parameters[:2], "lr": 5e-5},
        {"params": parameters[2:], "lr": 2e-5},
    ], group_grad_clip_norm=1.0, betas=(0.9, 0.95), eps=1e-6, weight_decay=0.01)


reference = make(MolmoAct2AdamW, reference_parameters)
offloaded = make(CPUStateMolmoAdamW, offloaded_parameters)


def check_equal():
    for original, candidate in zip(reference_parameters, offloaded_parameters):
        torch.testing.assert_close(original, candidate, atol=0, rtol=0)
        for name, value in reference.state[original].items():
            actual = offloaded.state[candidate][name]
            assert actual.device.type == "cpu"
            torch.testing.assert_close(value.cpu(), actual, atol=0, rtol=0)


for step in range(8):
    for index, (original, candidate) in enumerate(zip(reference_parameters, offloaded_parameters)):
        gradient = torch.randn_like(original)
        original.grad = gradient.clone()
        candidate.grad = gradient.clone()
        if step == 3 and index == 2:
            original.grad = candidate.grad = None
    reference.step()
    offloaded.step()
    check_equal()
    reference.zero_grad(set_to_none=True)
    offloaded.zero_grad(set_to_none=True)
    if step == 4:
        # Exercise loading normal CUDA Adam states into CPU-state Adam.
        offloaded.load_state_dict(copy.deepcopy(reference.state_dict()))
        check_equal()
    if step == 6:
        # Also resume serialized CPU moments, as produced by an offloaded run.
        checkpoint = io.BytesIO()
        torch.save(offloaded.state_dict(), checkpoint)
        checkpoint.seek(0)
        offloaded = make(CPUStateMolmoAdamW, offloaded_parameters)
        offloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
        check_equal()

before = [p.detach().clone() for p in offloaded_parameters]
for original, candidate in zip(reference_parameters, offloaded_parameters):
    original.grad = torch.full_like(original, float("nan"))
    candidate.grad = torch.full_like(candidate, float("nan"))
reference.step()
offloaded.step()
check_equal()
assert all(torch.equal(old, current) for old, current in zip(before, offloaded_parameters))
print(json.dumps({
    "status": "passed", "updates": 8, "exact_parameter_and_moment_equality": True,
    "multiple_groups_and_learning_rates": True, "weight_decay": True,
    "missing_gradient": True, "nonfinite_step_skipped": True,
    "native_adam_checkpoint_load": True, "all_moments_on_cpu": True,
    "serialized_cpu_checkpoint_resume": True,
    "torch": torch.__version__,
}, indent=2))
