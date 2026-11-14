# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Object colour DR, driven by ``[task.dr] object_colors``; chosen names are kept per environment.

``env._object_colors[object_name][index]`` holds the colour name last applied to that environment's
object (None until one is applied); the task's episode metadata reports it.
"""
import torch
from pxr import Gf, Usd, UsdShade

from isaaclab.sim import get_current_stage

from sim.workspace.envs.mdp._common import _env_prim_path, env_indices


def randomize_object_color(env, env_ids, colors, object_name):
    """Pick a random palette colour per env, apply it, and return the chosen names."""
    env_ids = env_indices(env, env_ids)
    names = list(colors)
    if not names:
        raise ValueError("Object colour palette must not be empty")
    selected_names = []
    for index in env_ids.tolist():
        name = names[torch.randint(len(names), (1,)).item()]
        set_object_color(env, torch.tensor([index], device=env.device), colors[name], name, object_name)
        selected_names.append(name)
    return selected_names


def set_object_color(env, env_ids, rgb: tuple[float, float, float], name, object_name):
    """Set the object's bound surface shader colour in each listed environment."""
    env_ids = env_indices(env, env_ids)
    asset = env.scene[object_name]
    stage = get_current_stage()
    if not hasattr(env, "_object_colors"):
        env._object_colors = {}
    applied = env._object_colors.setdefault(object_name, [None] * env.num_envs)

    for index in env_ids.tolist():
        path = _env_prim_path(env, asset, index)  # v1: asset.root_physx_view.prim_paths[index]
        # Newton's scene topology and material bindings are fixed after startup.
        # Cache the resolved inputs on the env, not globally across USD stages.
        cache_stage, inputs = getattr(env, "_object_color_inputs", (None, {}))
        if cache_stage != stage:
            inputs = {}
            env._object_color_inputs = (stage, inputs)
        if path not in inputs:
            root = stage.GetPrimAtPath(path)
            shaders = {}
            for prim in Usd.PrimRange(root):
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if material:
                    shader, _, _ = material.ComputeSurfaceSource()
                    if shader:
                        shaders[str(shader.GetPath())] = shader
            if not shaders:
                raise RuntimeError(f"No bound surface shader found for {object_name} at {path}")
            resolved = []
            for shader in shaders.values():
                diffuse = shader.GetInput("diffuseColor")
                if not diffuse:
                    raise RuntimeError(f"{object_name} shader {shader.GetPath()} has no diffuseColor input")
                resolved.append(diffuse)
            inputs[path] = tuple(resolved)
        for diffuse in inputs[path]:
            diffuse.Set(Gf.Vec3f(*rgb))
        applied[index] = name
