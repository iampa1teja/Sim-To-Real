# Generalist VLA action spaces for zero-shot use on a new 7-DoF arm (SP7) through IK

Scope: open-weight generalist robot policies whose output can be expressed as end-effector (EE) commands that the SP7 differential-IK layer could execute. The target is a 7-revolute arm with a parallel prismatic gripper, one wrist camera, one fixed third-person camera, Isaac Lab 3.0 sim plus Feetech hardware, and no demonstrations. Research date: 2026-11-11. Primary sources are HF model cards and files, fetched via the HF API and `resolve/main` URLs, plus official GitHub READMEs and configs. Anything not read from a primary source in this session is marked **[unverified]**.

## Q1. Weights, licences, hosting, and per-checkpoint action spaces

### Takeaway
Few released checkpoints output EE actions that an IK layer can consume:
- The OXE/Bridge/Fractal-trained 7-D delta-EE models: OpenVLA, Octo, SpatialVLA, CogACT, MolmoAct v1, Magma, and the delta-EE tags inside MolmoAct2.
- GR00T N1.7's relative-EEF (xyz + 6D rotation) DROID tag.
- LAP-3B (2026), which uses delta-EE.

All the openpi DROID checkpoints, and MolmoAct2's DROID and SO100 heads, output joint-space actions. SmolVLA and the LIBERO fine-tunes are not usable zero-shot on a new arm.

### Cited Findings

**Comparison table.** Each cell was checked against the source in its row.

| Model / checkpoint (HF id) | Licence (HF tag / card) | Output action space | Rotation / gripper | Chunk / horizon | Cameras / proprio | Source |
|---|---|---|---|---|---|---|
| `openvla/openvla-7b` | MIT | 7-DoF action, un-normalized per dataset via `unnorm_key` (e.g. `bridge_orig`) | 7-D = EE delta + gripper per OXE convention **[delta-EE form unverified in README; standard OpenVLA knowledge]** | 1 step | 1 third-person image, no proprio **[unverified]** | [OpenVLA README](https://github.com/openvla/openvla) |
| `moojink/openvla-7b-oft-finetuned-libero-*` (OpenVLA-OFT) | MIT | Chunk of actions, un-normalized with `libero_spatial_no_noops`; LIBERO-only fine-tune | LIBERO delta-EE **[unverified]** | `NUM_ACTIONS_CHUNK` constant | `use_proprio=True`; wrist + third-person in LIBERO config | [OFT README](https://github.com/moojink/openvla-oft) |
| `rail-berkeley/octo-base-1.5` (also octo-small-1.5) | MIT | Action chunk of 4; history window 2 | Dataset-specific (Bridge: delta EE) **[unverified detail]** | 4 | "multiple RGB camera inputs" | [Octo README](https://github.com/octo-models/octo) |
| `IPEC-COMMUNITY/spatialvla-4b-224-pt` | MIT | Actions decoded with `unnorm_key` (e.g. `bridge_orig/1.0.0`) | OXE 7-D delta EE **[unverified]** | n/a | 1 image; ~8.5 GB GPU | [SpatialVLA card](https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt) |
| `CogACT/CogACT-Base` | MIT | "next 16 normalized robot actions (7-DoF end effector deltas `x, y, z, roll, pitch, yaw, gripper`)", unnormalized with e.g. `fractal20220817_data`; optional Adaptive Action Ensemble | Euler RPY delta + gripper | 16 (`future_action_window_size=15`) | "single view RGB image" | [CogACT card](https://huggingface.co/CogACT/CogACT-Base) |
| `robotics-diffusion-transformer/rdt-1b` | MIT | 128-D "unified action space" that includes joint and EE fields; you fill the slots your robot has | EE: `eef_pos` (idx 30–32) + "6D pose" `eef_angle_0..5` (idx 33–38); gripper `gripper_open` idx 10 | 64 actions | Up to 3 views, last 2 frames; requires proprio and a `control_frequency` input (example 25) | [RDT card](https://huggingface.co/robotics-diffusion-transformer/rdt-1b), [state_vec.py](https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/main/configs/state_vec.py) |
| `microsoft/Magma-8B` | MIT | OXE "7 DoF robot gripper state" actions; evaluated zero-shot on SimplerEnv and a real WidowX | OXE 7-D **[unverified detail]** | n/a | n/a | [Magma card](https://huggingface.co/microsoft/Magma-8B) |
| `allenai/MolmoAct-7B-D-0812` (+ `-Pretrain-0812`, `-O-0812`, LIBERO variants) | Apache-2.0 | Text-parsed actions, `parse_action(..., unnorm_key="molmoact")`; trained on an OXE subset + 10k Franka trajectories | Delta EE **[unverified for v1 card; matches MolmoAct2 `franka_molmoact` tag below]** | n/a | 2 images in example | [MolmoAct card](https://huggingface.co/allenai/MolmoAct-7B-D-0812) |
| `allenai/MolmoAct2` (base, post-trained multi-embodiment) | No licence tag on HF repo | Per-tag `control_mode` in `norm_stats.json` (see list below); `norm_mode: q01_q99` | See tags | 3–30 per tag | See tags | [norm_stats.json](https://huggingface.co/allenai/MolmoAct2/blob/main/norm_stats.json) |
| `allenai/MolmoAct2-DROID`, `-SO100_101`, `-BimanualYAM`, `-LIBERO`, `-Think`, `-Pretrain`, `-FAST-Tokenizer` | No licence tag (DROID repo) | Fine-tuned heads; the DROID tag in the base is joint-space | — | — | — | [HF API listing](https://huggingface.co/api/models?author=allenai&search=MolmoAct) |
| `nvidia/GR00T-N1.7-3B` (base) | NVIDIA Open Model License (commercial OK); code Apache-2.0 | Pretrain tag `oxe_droid_relative_eef_relative_joint`: action = `eef_9d` RELATIVE in `XYZ_ROT6D` format + `gripper_position` ABSOLUTE + `joint_position` RELATIVE | xyz + 6D rotation, relative to current pose; gripper absolute | 40 steps (`delta_indices` 0..39); recommended execution horizon 8–16 | Video `exterior_image_1_left` + `wrist_image_left` at delta indices [-15, 0]; state `eef_9d`, `gripper_position`, `joint_position`; `state_dropout_prob 0.2` | [processor_config.json](https://huggingface.co/nvidia/GR00T-N1.7-3B/blob/main/processor_config.json), [policy.md](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md), [card](https://huggingface.co/nvidia/GR00T-N1.7-3B) |
| `nvidia/GR00T-N1.7-DROID`, `-LIBERO`, `-SimplerEnv-Bridge`, `-SimplerEnv-Fractal` | NVIDIA Open Model License | Post-train tags `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`, `LIBERO_PANDA`, `SIMPLER_ENV_WIDOWX`, `SIMPLER_ENV_GOOGLE` | — | — | — | [Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T) |
| `nvidia/GR00T-N1.6-3B` | Card links "NVIDIA-OneWay-Noncommercial-License" | Per-embodiment MLP decoders; "continuous-value vectors correspond to different motor controls" | — | — | "any resolution" RGB, proprio required | [N1.6 card](https://huggingface.co/nvidia/GR00T-N1.6-3B) |
| openpi `pi0_base`, `pi0_fast_base`, `pi05_base` (GCS `gs://openpi-assets/checkpoints/...`; LeRobot ports `lerobot/pi0_base`, `lerobot/pi05_base`) | `license:gemma` on LeRobot ports | "Fine-Tuning" use case only | — | — | — | [openpi README](https://github.com/Physical-Intelligence/openpi) |
| openpi `pi0_fast_droid`, `pi0_droid`, `pi05_droid` | as above | DROID outputs `actions[..., :8]` (7 joint + gripper). `LeRobotDROIDDataConfig` comment: "We assume joint *velocity* actions"; "The original pi0.5-DROID model was trained with joint velocity actions" | Joint-space, not EE | `pi0_droid` horizon 10; `pi0_fast_droid` action_dim 8, horizon 10 | `exterior_image_1_left` + `wrist_image_left` 224×224, state = 7 joint pos + gripper | [droid_policy.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/droid_policy.py), [config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py), [README_train](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md) |
| openpi `pi05_libero` | as above | LIBERO fine-tune ("Inference") | — | — | — | [openpi README](https://github.com/Physical-Intelligence/openpi) |
| `lerobot/smolvla_base` | Apache-2.0 | "continuous actions"; fine-tune flow with `select_action` + user postprocess/unnormalize | SO100-family joint data **[unverified]** | — | — | [SmolVLA card](https://huggingface.co/lerobot/smolvla_base) |
| `lihzha/LAP-3B` (2026), `lihzha/LAP-3B-Libero` | Apache-2.0 (HF tag) | Delta EE poses; translations in cm, Euler rotations (right-hand rule); language-action convention +x forward, +y left, +z up | Euler; gripper is a binary proprio input (action output form not stated) | 25 Hz inference; chunk H not stated | Up to 2 images at 224×224; proprio = Cartesian EE pose (pos + continuous 6D rot + binary gripper) as text, 255 bins | [LAP paper](https://arxiv.org/html/2602.10556v1), [LAP README](https://github.com/lihzha/lap) |

**MolmoAct2 base `norm_stats.json` `metadata_by_tag`.** Read directly from the file. Format `molmoact2_norm_stats.v1`, `norm_mode: q01_q99`. Source: [norm_stats.json](https://huggingface.co/allenai/MolmoAct2/blob/main/norm_stats.json).
- `widowx_bridge`
  - `control_mode: "delta end-effector pose"`, action_horizon 5.
  - Action names `[x, y, z, roll, pitch, yaw, gripper]`; mask `[T×6, F]`, so the gripper is not normalized.
  - Cameras `image_0..image_3`.
  - Action q01/q99 for x is about ±0.029.
  - State has 8 dims.
- `google_robot_fractal`: `delta end-effector pose`, horizon 3, 1 camera (`observation.images.image`).
- `franka_molmoact`: `delta end-effector pose`, `action_key: action.del_ee_action`, horizon 10, cameras `primary` and `secondary`. Action std for xyz is about 0.005–0.008.
- `google_robot_bc_z`: `delta end-effector pose`, horizon 10, 1 camera.
- `franka_droid`
  - `control_mode: "absolute joint pose"`, horizon 15.
  - Cameras: `exterior_1_left`, `exterior_2_left`, `wrist_left`.
  - Action dims 0–6 hold joint angles in radians (e.g. dim 3 ranges −2.95…−0.19); the gripper is in [0, 1].
- `so100_so101_molmoact2`: `absolute joint pose`, horizon 30, `normalize_gripper: true`. Values look like degrees (min −270, max 269.8). The `camera_keys` list is empty.
- `yam_dual_molmoact2`: `absolute joint pose`, 14-D bimanual, horizon 30, cameras `top`, `left`, `right`.

**Other licence and backbone facts:**
- The MolmoAct2 base card says the checkpoint is "a foundation checkpoint for further robot fine-tuning rather than as a ready-to-run policy". It intentionally gives no inference code and points to the fine-tuned checkpoints instead. — [MolmoAct2 card](https://huggingface.co/allenai/MolmoAct2)
- The MolmoAct2 paper says the authors release "model weights, training code, and complete training data". The HF repos `allenai/MolmoAct2` and `allenai/MolmoAct2-DROID` carry no licence tag. — [arXiv 2605.02881](https://arxiv.org/abs/2605.02881); HF API (no `license:` tag returned)
- GR00T N1.7 uses `nvidia/Cosmos-Reason2-2B`, a gated model, as its backbone. Every N1.7 checkpoint needs HF access approval for it. — [Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T)

### Inferences
- Checkpoints that emit EE-space actions and could therefore be driven through SP7's IK:
  - OXE delta-EE heads (Bridge/WidowX or Fractal/Google-robot unnorm keys): OpenVLA, Octo, SpatialVLA, CogACT, Magma, MolmoAct v1, and the MolmoAct2 base `widowx_bridge`, `google_robot_*` and `franka_molmoact` tags.
  - GR00T N1.7 base with the DROID relative-EEF tag. Use only the `eef_9d` and gripper slices and discard the relative `joint_position` slice.
  - LAP-3B.
  - RDT-1B, if you fill only the EE slots.
- openpi DROID checkpoints (π0, π0-FAST, π0.5), MolmoAct2's `franka_droid` and SO100 tags, and the MolmoAct2-DROID fine-tune are joint-space. SP7's 7-joint kinematics differ from Franka's, so these cannot drive SP7 directly. A Franka FK→SP7 IK retargeting step is conceivable, but it is untested and the joint-space priors depend on Franka's morphology.
- GR00T N1.7's DROID action also carries a relative 7-joint delta. Because SP7 has 7 joints, the vector would technically fit, but those joint semantics are Franka's. Use the EEF channel.

### Gaps
- I did not read the OpenVLA, Octo, SpatialVLA or Magma action-dimension semantics (delta EE frame, Euler order) from primary files in this session. Their "7-D delta EE + gripper" form is the well-known OXE/Bridge convention, but it is **[unverified here]**.
- MolmoAct2 licence: no HF licence tag. Check the GitHub repo LICENSE.
- GR00T N1.5 embodiment tags and action spaces were not read. I believe N1.5 had `OXE_DROID`/`GR1` pretrain tags, but this is **[unverified]**.
- The pi0-FAST-DROID action type was not confirmed separately from pi0.5-DROID. The training README says the pi-DROID models used joint velocity.
- No primary source was read for SmolVLA's action space. Its base is SO100 community data, so it is joint-space **[unverified]**.

## Q2. Frames, rotation parameterization, gripper and normalization details relevant to IK

### Takeaway
Frames and units differ between models and must be matched per checkpoint:
- **OXE delta-EE models:** Euler-RPY deltas, un-normalized per dataset (Bridge vs Fractal stats differ by roughly 3–10×).
- **GR00T N1.7:** xyz + 6D rotation relative to the current EE pose, with absolute gripper position, using percentile normalization from the checkpoint's statistics.json.
- **LAP:** centimetres and Euler angles in a +x-forward / +y-left / +z-up language frame.

None of the model cards documents the frame (base vs. EE vs. camera) precisely enough to apply to a new arm without empirical checking.

### Cited Findings
- CogACT outputs `x, y, z, roll, pitch, yaw, gripper` deltas, unnormalized with dataset statistics (`unnorm_key`). Its optional Adaptive Action Ensemble also "depend[s] on the dataset statistics". — [CogACT card](https://huggingface.co/CogACT/CogACT-Base)
- MolmoAct2 un-normalizes with q01/q99 statistics per tag, and the gripper is masked out of normalization for the delta-EE tags. Per-tag scales differ a lot:
  - Bridge xyz q99 is about 0.028–0.041.
  - `franka_molmoact` xyz std is about 0.005–0.008.
  - Fractal xyz std is about 0.06–0.07.
  - Source: [norm_stats.json](https://huggingface.co/allenai/MolmoAct2/blob/main/norm_stats.json)
- In the MolmoAct2 Bridge tag, the gripper q01 is about 0 and q99 is about 1 (mean 0.58), so the gripper is in [0, 1]. Fractal's gripper mean is 0.535. Neither the file nor the card says which value means open. — [norm_stats.json](https://huggingface.co/allenai/MolmoAct2/blob/main/norm_stats.json)
- GR00T N1.7 processor settings:
  - Action configs: `{rep: RELATIVE, type: EEF, format: XYZ_ROT6D, state_key: eef_9d}` and `{rep: ABSOLUTE, type: NON_EEF, state_key: gripper_position}`.
  - `use_percentiles: true`, `use_relative_action: true`, `max_action_horizon 40`.
  - Images are cropped to 230 and resized to 256.
  - Source: [processor_config.json](https://huggingface.co/nvidia/GR00T-N1.7-3B/blob/main/processor_config.json)
- GR00T N1.7 describes the relative EEF action space as "deltas from the current pose (rather than absolute targets)". It says this improves generalization and is "a key factor in the model's cross-embodiment performance". The same space is shared with 20K hours of EgoScale human video. — [Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T)
- RDT-1B uses absolute EE position plus a 6D orientation in fixed slots of its 128-D unified vector. "To deploy on your robot platform, you need to fill the relevant quantities of the raw action vector into the unified space vector." — [RDT card](https://huggingface.co/robotics-diffusion-transformer/rdt-1b), [state_vec.py](https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/main/configs/state_vec.py)
- LAP: "translations in centimeters, and rotations as Euler angles (right-hand rule)", with a language-action convention of +x forward, +y left, +z up. — [LAP paper](https://arxiv.org/html/2602.10556v1)
- Cloak (CoRL 2026) retargets state and actions between source and target embodiments through tip poses. It computes target actions by IK on corresponding fingertip poses and masks the end-effector from the VLA's images. It is fine-tuned from π0.5 on one parallel-jaw source embodiment. — [Cloak arXiv 2606.22836](https://arxiv.org/abs/2606.22836); I did not verify whether weights are released.

### Inferences
- For any OXE delta-EE model on SP7, the user has to choose:
  - Which `unnorm_key` to apply. Bridge/WidowX has small 5 Hz-style steps; Fractal has larger steps.
  - Which frame to apply the deltas in. Bridge deltas are commonly treated as base-frame translation plus a rotation delta, but SimplerEnv implementations add per-robot frame conversions; **[unverified]**.
  - The Euler composition order.
  - The gripper polarity mapping onto SP7's prismatic stroke.
- These choices interact with the IK step size. A wrong `unnorm_key` scales motion by about 3–10×.
- GR00T's relative-EEF with 6D rotation is the cleanest fit for Isaac Lab's differential IK in "pose relative" mode. Before a quaternion goes into the Isaac Lab 3.0 IK (XYZW), convert the 6D rotation to a rotation matrix and then to a quaternion. Per repo conventions, keep XYZW internally and WXYZ only at the dataset boundary.
- The policy also needs proprio state `eef_9d`: SP7's FK pose given as xyz + 6D, in a frame that resembles DROID's Franka base frame.

### Gaps
- The exact definition of the GR00T N1.7 `eef_9d` frame is not stated in the files read: base vs. world, the 6D column convention, and whether relative means delta = T_cur⁻¹·T_target or a world-frame difference. It needs reading `gr00t_n1d7` code.
- MolmoAct2 and OXE delta-EE frame definitions (base vs. EE-local rotation delta) are not documented in the cards read.
- Gripper sign conventions (1 = open or closed) were not stated in any primary file read.

## Q3. Camera and proprio expectations

### Takeaway
SP7's layout of one third-person camera plus one wrist camera matches the DROID-style two-view input of GR00T N1.7's DROID tag, openpi DROID and MolmoAct2 franka_droid. Bridge/Fractal-style models expect a single third-person image, so the wrist camera would go unused. GR00T, RDT, LAP and OFT need proprio state.

### Cited Findings
- GR00T N1.7 DROID tag: video keys `exterior_image_1_left` and `wrist_image_left`, each at time offsets [-15, 0] (two frames). State is `eef_9d`, `gripper_position` and `joint_position`. — [processor_config.json](https://huggingface.co/nvidia/GR00T-N1.7-3B/blob/main/processor_config.json)
- openpi DROID: `exterior_image_1_left` and `wrist_image_left`, 224×224, plus 7 joint positions and the gripper. — [droid_policy.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/droid_policy.py)
- openpi DROID deployment: "You _do not_ need to carefully control camera angle, object positions, etc." — [openpi DROID example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README.md)
- CogACT takes a "single view RGB image". — [CogACT card](https://huggingface.co/CogACT/CogACT-Base)
- Octo was trained with a history window of 2 and supports multiple RGB cameras. — [Octo README](https://github.com/octo-models/octo)
- RDT takes up to 3 views, the last 2 frames, proprio, and a control frequency. — [RDT card](https://huggingface.co/robotics-diffusion-transformer/rdt-1b)
- LAP takes up to 2 images at 224×224 and requires Cartesian EE proprio. Its DROID deploy script takes external and wrist camera ids. — [LAP paper](https://arxiv.org/html/2602.10556v1), [LAP README](https://github.com/lihzha/lap)
- MolmoAct2 tag cameras: Bridge 4 slots, Fractal 1, `franka_molmoact` 2 (`primary`, `secondary`), DROID 3. — [norm_stats.json](https://huggingface.co/allenai/MolmoAct2/blob/main/norm_stats.json)

### Inferences
- With only one fixed camera, SP7 can feed the "exterior" slot. Camera pose relative to the robot base should roughly resemble the training viewpoint (DROID: side-mounted ZED cameras; Bridge: over-the-shoulder).
- The SP7 wrist camera has to be mounted to resemble the Franka/DROID wrist view for it to help.

### Gaps
- None of the sources read here quantifies how sensitive EE-space models are to the robot's appearance in the image. Cloak's method, masking the end effector, suggests this sensitivity is a major failure source; that is an inference from its design.

## Q4. Documented zero-shot results on unseen robots or new setups

### Takeaway
Primary sources consistently say that pre-LAP OXE VLAs and openpi checkpoints work "out of the box" only on embodiments in their training mix. The 2026 LAP paper reports that π0.5-DROID, π0.5-Base, X-VLA, MolmoAct and OpenVLA all collapse to 0% on unseen robots, while LAP-3B reaches over 50% average success. GR00T N1.7 advertises zero-shot use only for its pretrain tags.

### Cited Findings
- OpenVLA: "typically requires fine-tuning on a small demonstration dataset (~100 demos) from your target domain robot. Out-of-the-box, it only works well on domains from the training dataset." — [OpenVLA README](https://github.com/openvla/openvla)
- SpatialVLA: "do not zero-shot generalize to new (unseen) robot embodiments, or setups that are not represented in the pretraining mix". — [SpatialVLA card](https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt)
- SpatialVLA SimplerEnv Google-robot table, final average column: SpatialVLA zero-shot 68.8% vs RoboVLM zero-shot 46.3%. This is an in-distribution robot (Google robot is in OXE). The meaning of the columns was not verified. — [SpatialVLA card](https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt)
- CogACT "can be used zero-shot to control robots for setups seen in the Open-X pretraining mixture". — [CogACT card](https://huggingface.co/CogACT/CogACT-Base)
- Octo out of the box controls robots from its training set; new action spaces and morphologies need "a small target domain dataset" fine-tune. — [Octo README](https://github.com/octo-models/octo)
- openpi on its expert checkpoints: "These may or may not work on your particular robot." pi0-FAST-DROID "can perform a wide range of simple table-top manipulation tasks 0-shot in new scenes on the DROID robot platform". That claim covers a new scene on the same robot, not a new robot. — [openpi README](https://github.com/Physical-Intelligence/openpi)
- Magma reports zero-shot SimplerEnv and real WidowX evaluation. WidowX is in its OXE training data. — [Magma card](https://huggingface.co/microsoft/Magma-8B)
- GR00T N1.7:
  - "Pretrain tags ... use for zero-shot inference on datasets that match the pretrained embodiment."
  - "`NEW_EMBODIMENT` ... requires `--modality-config-path` during finetuning."
  - Posttrain tags passed to the base model "will fail with an error".
  - Source: [policy.md](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md)
- LAP-3B, tested zero-shot on a custom 7-DoF Franka, the 6-DoF YAM and the 7-DoF Kinova:
  - LAP-3B: over 50% average success, about 2× the strongest baseline.
  - "open-sourced baselines (π0.5-DROID, π0.5-Base, X-VLA, MolmoAct, OpenVLA) all collapse to zero success".
  - Per-robot numbers are only in the paper's Fig. 3 and were not extracted.
  - Source: [LAP paper](https://arxiv.org/html/2602.10556v1)
- MolmoAct2 reports outperforming π0.5 across 7 simulation and real benchmarks. The abstract gives no unseen-robot zero-shot numbers. — [arXiv 2605.02881](https://arxiv.org/abs/2605.02881)
- Cloak is presented as "Zero-Shot Cross-Embodiment Manipulation by Masking the End-Effector from the VLA". It is built on π0.5 with tip-pose IK retargeting. — [Cloak](https://arxiv.org/abs/2606.22836), [project page](https://tml.stanford.edu/cloak/)

### Inferences
- Based on documented evidence, LAP-3B (`lihzha/LAP-3B`, Apache-2.0, delta EE, DROID-style deployment script) is the strongest open-weight candidate for zero-shot SP7 through IK.
- The second candidate is GR00T N1.7 base with the DROID relative-EEF tag. It runs zero-shot, but it was only documented on DROID-matching data.
- The OXE delta-EE models make useful baselines and will probably be weak. The LAP paper's 0% for OpenVLA and MolmoAct on unseen arms supports this.
- SP7's 7-revolute layout may make Franka/DROID-trained EE policies more plausible than for 6-DoF arms. This is speculative.

### Gaps
- No primary source was found that reports Octo, RDT, CogACT or GR00T zero-shot success on a robot outside their training set.
- LAP per-robot numbers and gripper output convention: read Fig. 3 and the code.
- No independent replication of LAP's claims was found.

## Q5. Inference compute and latency

### Takeaway
All candidates fit on a single 16–24 GB consumer GPU at bf16:
- openpi: >8 GB.
- SpatialVLA: about 8.5 GB.
- OpenVLA-OFT: about 16–18 GB.
- GR00T N1.7: 16 GB or more, 12–36 Hz depending on the backend.
- LAP: 25 Hz on an RTX 4090.

### Cited Findings
- openpi inference needs more than 8 GB (RTX 4090). Remote websocket serving is supported. With the DROID setup, "0.5 - 1 sec latency per chunk is normal". — [openpi README](https://github.com/Physical-Intelligence/openpi), [DROID example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README.md)
- GR00T N1.7 compute and timing (4 denoising steps, 1 camera):
  - Inference on 1 GPU with 16 GB or more.
  - End-to-end latency: H100 eager 85.8 ms (11.7 Hz); TensorRT 27.9 ms (35.9 Hz).
  - L40: eager 128.3 ms; TRT 38.4 ms.
  - Orin: 216.5 ms at best.
  - First download is about 6 GB.
  - Sources: [N1.7 card](https://huggingface.co/nvidia/GR00T-N1.7-3B), [README](https://github.com/NVIDIA/Isaac-GR00T)
- OpenVLA-OFT inference: about 16 GB VRAM for LIBERO and about 18 GB for ALOHA. — [OFT README](https://github.com/moojink/openvla-oft)
- SpatialVLA needs 8.5 GB of GPU memory for inference. — [SpatialVLA card](https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt)
- OpenVLA: the 7B checkpoint download is "30 GB" (the README's fine-tuned checkpoint example). — [OpenVLA README](https://github.com/openvla/openvla)
- LAP: 25 Hz on an RTX 4090 (action expert rollout). — [LAP paper](https://arxiv.org/html/2602.10556v1)

### Inferences
- The repo's existing GR00T/Molmo serving path (Phase 4 policy serving) suggests GR00T N1.7 or MolmoAct2 slot into existing infrastructure most easily. LAP would need a new policy-folder client.

### Gaps
- MolmoAct2, CogACT, RDT-1B, Octo and Magma latency or VRAM were not found in the primary sources read.

## Q6. Known pitfalls when deploying on a different arm

### Takeaway
The main failure modes are:
- Picking the wrong normalization tag or `unnorm_key`.
- A mismatched EE frame or rotation convention, including relative vs. absolute and Euler order.
- Gripper polarity and range.
- Sending joint-space checkpoints to a non-matching arm.
- Visual embodiment gap, since the model sees an unfamiliar arm.

### Cited Findings
- GR00T requires the tag to match: "Must match the robot type of your dataset." You "cannot mix embodiment tags". — [policy.md](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md)
- The openpi training README warns that pi0.5-DROID used joint velocity actions, which "are not compatible with simulated evaluation environments (much harder to simulate)". — [README_train](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md)
- MolmoAct2 safety guidance: the action space "should be bounded by the training data, robot controller limits, and task-specific safety constraints". It recommends validating through simulation rollouts and action limits before running on hardware. — [MolmoAct2 card](https://huggingface.co/allenai/MolmoAct2)
- Delta-EE and joint-pose tags sit side by side in the MolmoAct2 base norm stats. Selecting the `franka_droid` tag yields absolute Franka joint angles, not EE. — [norm_stats.json](https://huggingface.co/allenai/MolmoAct2/blob/main/norm_stats.json)
- The Cloak paper attributes cross-embodiment failure partly to the VLA seeing a different end effector. Its remedy is to mask the end effector and retarget through IK on tip poses. — [Cloak](https://arxiv.org/abs/2606.22836)

### Inferences
Suggested checklist for SP7:
1. Map SP7's TCP and base frame onto the training robot's conventions (DROID Franka base, or Bridge WidowX base).
2. Convert rotations explicitly: Euler, 6D, or quaternion, keeping XYZW in Isaac Lab.
3. Clip per-step deltas to the q01/q99 bounds of the chosen tag.
4. Map gripper [0, 1] to the prismatic stroke in metres, verifying polarity in sim first.
5. Run in Isaac Lab before touching the Feetech hardware.

### Gaps
- No primary source compared unnormalization-stat choices on a new arm.
