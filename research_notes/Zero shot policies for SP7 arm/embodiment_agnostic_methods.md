# Embodiment-agnostic, zero-shot manipulation methods for the SP7 arm (excluding end-to-end robot-action VLAs)

Scope note: research done 2026-11-11 with about 20 search/fetch calls. The GitHub API was blocked in this session (HTTP 403), so repository details come from WebFetch of the GitHub HTML pages and project pages. Licence statements are only as reliable as those page renders. Where a claim says "page says", I could not confirm it in the code itself.

SP7 context assumed: 7-revolute-joint arm, two-finger parallel prismatic gripper, wrist RGB plus a fixed third-person RGB camera, sim depth available, working IK in Isaac Lab 3.0, no demonstrations and no leader arm. Target tasks: cube-in-box and stacking.

## Q1. Which methods have code (and weights) actually released, under what licence?

### Takeaway
Released and usable now: NVIDIA GraspGen-X (Apache-2.0 code, NVIDIA Open Model Licence weights, zero-shot on any gripper from a URDF), GraspGen (per-gripper models), VoxPoser (MIT, RLBench demo), ReKep (OmniGibson sim demo only, no real-world perception stack), DreamZero (Apache-2.0 code, 14B weights on HF), KITE (code and decoder weights, no licence file), Cloak (repo linked) and Im2Flow2Act (repo linked). AnyGrasp is a licence-gated binary SDK. Several 2026 papers have no confirmed code.

### Cited Findings
**Grasp foundation models (family 2)**
- GraspGen-X (CVPR 2026, arXiv 2606.00998) is one cross-embodiment 6-DoF diffusion grasp model conditioned on a swept-volume gripper representation (a 12-dim vector approximating the finger sweep). It targets novel grippers not seen in training — [arXiv 2606.00998](https://arxiv.org/html/2606.00998v1); [NVlabs/GraspGenX](https://github.com/NVlabs/GraspGenX)
- GraspGen-X code is Apache-2.0 and its checkpoints are under the NVIDIA Open Model License. Commercial deployment requires contacting NVIDIA. Weights are on Hugging Face (`adithyamurali/GraspGenXModel`), trained on "over 2 billion grasps" across 32 procedural grippers — [NVlabs/GraspGenX](https://github.com/NVlabs/GraspGenX)
- A third-party HF checkpoint `roboticascent/graspgenx-so101` exists for SO-101 (not verified further) — [HF](https://huggingface.co/roboticascent/graspgenx-so101)
- GraspGen (the original, ICRA '26) supports only Franka Panda, Robotiq-2F-140 and a 30 mm suction cup. Other grippers need retraining with a URDF, a gripper description and a grasp dataset, or retargeting for similar antipodal grippers via a z-offset. The README points to GraspGen-X for other grippers. Code release v1.0.0 was 2025-07-16. A ZMQ server and an MCP (LLM tool-calling) interface were added 2026-03-03. A Mixture-of-Experts model with top-down and side grasps was added 2026-06-21. Dataset is CC BY 4.0. The README render showed only "Copyright © 2025 NVIDIA… All rights reserved" plus LICENSE files whose text was not visible — [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)
- AnyGrasp is a binary SDK. Access requires applying for a licence tied to a machine feature ID. The old licence tool is deprecated, and applications with old feature IDs submitted before 4 July 2026 are still processed. Open issues report long waits (#175) and feature-ID mismatches after reboot (#178) — [anygrasp_sdk](https://github.com/graspnet/anygrasp_sdk); [issue #175](https://github.com/graspnet/anygrasp_sdk/issues/175); [issue #178](https://github.com/graspnet/anygrasp_sdk/issues/178); [NYU mirror note](https://github.com/NYU-robot-learning/anygrasp)
- The GraspGen authors could not compare against AnyGrasp in simulation because of its licence, and evaluated it only on a real robot. M2T2's contact-point formulation assumes symmetric, non-adaptive pinch grippers — [GraspGen paper](https://arxiv.org/html/2507.13097v1)

**VLM/LLM-planned methods (family 1)**
- VoxPoser: the official repo lists an MIT licence. It is a zero-shot RLBench demo implementation with `LMP.py` language-model programs and a playground notebook — [huangwl18/VoxPoser](https://github.com/huangwl18/VoxPoser); [project](https://voxposer.github.io/)
- ReKep: the repo is a demo in OmniGibson (pinned commit cc0316a, Isaac Sim 2023.1.1) with a pen-in-holder task. It "does not contain the perception pipeline used in our real-world experiments, including keypoint tracking, mask tracking, and SDF reconstruction"; the demo reads that information from simulation. It needs `OPENAI_API_KEY` (GPT-4o suggested). For real-world deployment the README suggests DINOv2, nvblox_torch, cuRobo, Cutie, Deoxys and AnyGrasp, but does not ship them. No licence was visible on the page — [huangwl18/ReKep](https://github.com/huangwl18/ReKep)
- OmniManip (CVPR 2025, PKU and AgiBot): a third-party write-up lists code at `pmj110119/OmniManip`, but the project page shows no code link and says "follow-up work will be released soon". I did not verify the repo — [arXiv 2501.03841](https://arxiv.org/pdf/2501.03841); [project](https://omnimanip.github.io)
- MOKA is a training-free mark-based visual prompting method. Search did not confirm a code repo — [arXiv 2403.03174](https://arxiv.org/pdf/2403.03174)
- "Bridging Semantics and Kinematics" (arXiv 2606.23157) is a training-free modular framework. A Molmo2 model routes the user's objective to predefined primitives such as pick-and-place and constrained push. It reports 62% end-to-end task success. No code was confirmed — [alphaXiv 2606.23157](https://www.alphaxiv.org/abs/2606.23157)
- Other 2026 VLM-planner papers surfaced but are unexamined: ZeroDex (dexterous hands, multi-view 3D-grounded VLM, arXiv 2606.19340), World Action Agent (arXiv 2609.29964) and In-Context Robot Learning with VLM Agents (arXiv 2609.19138) — [ZeroDex](https://arxiv.org/html/2606.19340v2); [WAA](https://arxiv.org/pdf/2609.29964); [ICRL-VLM](https://arxiv.org/pdf/2609.19138)

**Cross-embodiment / world-model / flow methods (family 3)**
- DreamZero (arXiv 2602.15922, NVIDIA GEAR): the code is Apache-2.0 at `dreamzero0/dreamzero`. Released checkpoints are DreamZero-DROID (14B inference checkpoint), DreamZero-AgiBot (~45 GB pretrained base for fine-tuning on new embodiments) and DreamZero-DROID-Data (~131 GB). It builds on Wan2.1-I2V-14B. The README does not state separate weight licence terms — [dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero); [project](https://dreamzero0.github.io/)
- KITE (arXiv 2606.22113): the repo `helloqxwang/KITE` has code (decoder training, viewers, Objects Placing pipeline) and a pretrained intent encoder plus decoder bank (`checkpoints/decoder/release.pt`). It runs in MuJoCo. It has no LICENSE file and a single commit. The page does not mention released interaction-policy weights, and the ~84 GB synthetic data is generated on first run — [helloqxwang/KITE](https://github.com/helloqxwang/KITE); [project](https://kite-manip.github.io/)
- Cloak (arXiv 2606.22836, Stanford TML): the project page links `github.com/Stanford-TML/cloak` and a setup guide. Weights are not mentioned. I did not inspect the repo or its licence — [project](https://tml.stanford.edu/cloak/); [arXiv](https://arxiv.org/abs/2606.22836)
- SkelWAM (arXiv 2609.21983): the arXiv page lists only a project page, which returned HTTP 503 when fetched. No code or weights were confirmed — [arXiv](https://arxiv.org/abs/2609.21983)
- Im2Flow2Act (CoRL 2024): code is at `real-stanford/im2Flow2Act` per the project page — [project](https://im-flow-act.github.io)
- Track2Act and Gen2Act: project pages exist, but search did not confirm code repos — [Track2Act](https://homangab.github.io/track2act/); [Gen2Act](https://homangab.github.io/gen2act/)
- Dream2Flow (arXiv 2512.24766, Stanford: Dharmarajan, W. Huang, Wu, Fei-Fei, R. Zhang): the arXiv page lists only a project website and no code repo — [arXiv](https://arxiv.org/abs/2512.24766)

### Inferences
- For SP7 the only components that are both released and truly zero-shot on a new parallel gripper are GraspGen-X (grasp poses) and the training-free VLM planners (VoxPoser, and ReKep with your own perception). The family-3 methods either need source-robot training (KITE, Cloak, SkelWAM) or fine-tuning data on the new robot (DreamZero).
- AnyGrasp's licence gating and its incompatibility with simulator benchmarking make it a poor fit for a sim-first Isaac Lab workflow. GraspGen-X is the natural substitute.

### Gaps
- Contact-GraspNet and M2T2 licence terms: not verified (the GitHub API was blocked and search did not return their LICENSE files).
- CoPa, MOKA, Code-as-Policies, Track2Act and Gen2Act code availability: not verified.
- DreamZero weight licence, Cloak repo licence and weights, OmniManip repo contents: not verified.
- MALMBot-like systems: I found no source matching that name.

## Q2. Inputs and outputs of each method

### Takeaway
Grasp models take segmented point clouds (depth plus camera calibration plus instance masks) and output ranked 6-DoF grasps. VLM planners take RGB-D plus language and output keypoint constraints, value maps or primitive parameters executed by IK or motion planning. DreamZero-style world-action models take multi-view RGB plus language and output joint/EE action chunks.

### Cited Findings
- GraspGen-X inputs are partial point clouds (e.g., from a depth camera), object meshes (obj/stl/ply/USD) or full scene point clouds with per-object segmentation. Output is 6-DoF grasp poses. The end-to-end pipeline exports USD for Isaac Sim/Omniverse — [GraspGenX](https://github.com/NVlabs/GraspGenX)
- For robot use, GraspGen says you need instance segmentation (e.g., SAM2) and a motion planner (e.g., cuRobo). Collision filtering uses a scene point cloud, and its demo takes depth and segmentation images — [GraspGen](https://github.com/NVlabs/GraspGen)
- ReKep's real-world version needs multiple RGB-D cameras plus keypoint tracking, mask tracking and SDF reconstruction. The robot interface `execute_action` receives a target end-effector pose and computes IK — [ReKep](https://github.com/huangwl18/ReKep)
- DreamZero is trained on three camera views and recommends aligning views in the same order as its training data. It outputs action chunks — [DreamZero repo](https://github.com/dreamzero0/dreamzero)
- Cloak masks the end-effector in the wrist view using robot state and wrist-camera parameters, and bridges embodiments by tip-pose retargeting through FK. A target therefore needs FK/IK over two tip frames, its geometry, and wrist camera parameters — [Cloak project](https://tml.stanford.edu/cloak/); [search summary of arXiv PDF](https://arxiv.org/pdf/2606.22836)
- SkelWAM uses a 25-D shared state (arm centreline geometry, TCP pose, parallel-jaw command) over third-person and wrist views. Embodiment-specific constrained decoders produce controls — [arXiv 2609.21983](https://arxiv.org/abs/2609.21983)
- Dream2Flow takes an initial image and a task instruction, reconstructs 3D object flow from a generated video, and converts it to commands via trajectory optimisation or RL — [arXiv 2512.24766](https://arxiv.org/abs/2512.24766)
- Im2Flow2Act generates object flow from human videos. A flow-conditioned policy trained on simulated play data executes it — [project](https://im-flow-act.github.io)

### Inferences
- SP7's fixed third-person camera in sim (with depth) is enough for the GraspGen-X plus VLM pipeline. On the real robot, a calibrated depth camera (third-person, or wrist RGB-D) is effectively mandatory for any point-cloud grasp model or 3D keypoint planner.
- Planner outputs (EE poses or waypoints) map directly onto SP7's existing IK controller and the task-folder `expert.py` waypoint style already in this repo.

### Gaps
- Exact camera-count and depth requirements for VoxPoser, MOKA and OmniManip were not re-verified from primary sources in this pass.

## Q3. What does adapting to a new arm (SP7) require?

### Takeaway
GraspGen-X needs only the gripper URDF and meshes plus a wizard-generated config. VLM planners need an EE-pose IK interface and camera calibration. DreamZero needs about 30 minutes of SP7 play data plus multi-GPU fine-tuning. KITE, Cloak and SkelWAM need only the target kinematic model or geometry, but also a policy trained on a source robot for the same task, which must come from somewhere.

### Cited Findings
- GraspGen-X: a new gripper needs a URDF and a `config.json` (open/close joint states, swept-volume boxes) generated by `scripts/gripper_config_wizard.py`. A `parallel_2f` gripper type exists, and the README says it runs zero-shot on any gripper given URDF plus meshes — [GraspGenX](https://github.com/NVlabs/GraspGenX); [search summary](https://github.com/NVlabs/GraspGenX)
- NVIDIA says GraspGen-X eliminates per-gripper training cycles — [NVIDIA blog](https://blogs.nvidia.com/blog/cvpr-research-grasping-driving-agent-training/)
- ReKep on a real robot: the README example is Franka. Most changes go in `environment.py`'s exposed functions, and `execute_action` takes a target EE pose, runs IK and sends joint commands — [ReKep](https://github.com/huangwl18/ReKep)
- DreamZero: adapting to the YAM arm used 30 minutes of play data (55 trajectories) with no other YAM data. The repo says post-training on about 30 minutes of play data "can enable basic language following and pick-and-place on a specific robot". The workflow is to convert a dataset per `docs/DATASET_TO_GEAR_AND_TRAIN.md` and then run LoRA or full fine-tuning from DreamZero-AgiBot. The abstract describes embodiment transfer as few-shot — [project](https://dreamzero0.github.io/); [repo](https://github.com/dreamzero0/dreamzero); [arXiv](https://arxiv.org/abs/2602.15922)
- KITE: the target needs only its kinematic model. A new embodiment adds input/output projectors and fine-tunes the decoder bank on synthetic kinematic data while the interaction policies stay frozen. For Wuji this took 50 epochs (~10 h on one RTX 5090). The interaction policy, however, is trained on demonstrations from a source robot — [project](https://kite-manip.github.io/); [arXiv](https://arxiv.org/abs/2606.22113)
- KITE's README gives no documented procedure for adding a new embodiment — [repo](https://github.com/helloqxwang/KITE)
- Cloak: "No data of new embodiments is ever collected". The VLA is trained on a single parallel-jaw (Robotiq) dataset — [arXiv](https://arxiv.org/abs/2606.22836); [project](https://tml.stanford.edu/cloak/)
- SkelWAM: no target-task demonstrations and no target policy updates. It needs an embodiment-specific constrained decoder, and the abstract does not say how that decoder is built — [arXiv](https://arxiv.org/abs/2609.21983)
- Im2Flow2Act: the policy is trained on about 4,800 simulated, task-less play samples. The flow generator is trained on action-less human videos — [project](https://im-flow-act.github.io)

### Inferences
- SP7 has no demonstrations, so KITE, Cloak and SkelWAM are "zero-shot on the target" only if a compatible source policy for cube pick/stack exists. KITE's released assets cover a MuJoCo Objects Placing task. Cloak's source data is a Robotiq dataset whose name was not given. In practice they would need source demonstrations, which the repo could generate with its own scripted `expert.py` on a different arm, defeating much of the purpose.
- DreamZero's 30-minute play-data requirement could plausibly be met with scripted or IK-generated sim play on SP7, but it is not demonstration-free in the strict sense. Its compute (≥2 GPUs, H100/GB200) is heavy.
- Im2Flow2Act's "sim play data" requirement fits an Isaac Lab setup (scripted random play), but its policy would need retraining for SP7.

### Gaps
- Whether Cloak's or KITE's released weights include a parallel-gripper pick/place source policy reusable for SP7: not verified.
- SkelWAM decoder construction effort: unknown (project page down).

## Q4. Reported zero-shot success on pick-and-place-style tasks; robots; sim or real

### Takeaway
Numbers exist mainly for the family-3 papers. They are mostly measured on hands and grippers other than a 7-DoF arm with a prismatic gripper, and are not directly comparable. None of the sources found here reports results on an SP7-like custom arm.

### Cited Findings
- KITE sim: 86.4% average across 12 task-target settings (e.g., Objects Placing on Barrett 74.7%, Allegro 84.0%, Wuji 80.0%). Real: Wuji hand on a Tianji Marvin arm, Objects Placing from a Robotiq source 8/10. It degrades under large initial-pose shifts (66% when flipped 180° vs 85%) — [KITE project](https://kite-manip.github.io/)
- Cloak average task progression (not success): source Robotiq ~88%. Zero-shot targets: UMI gripper 85%, YAM arm 86%, Sharpa hand 82%. On the Sharpa hand, the π0.5-droid baseline scored 54% and LAP-VLA 60% — [Cloak project](https://tml.stanford.edu/cloak/)
- SkelWAM: Franka-trained, 43.3% success over 1,000 episodes on LIBERO-Cross10 (sim, 10 target embodiments), +36.2 pp over the best baseline. Real-world results on the Feagine A03 continuum robot are qualitative only — [arXiv](https://arxiv.org/abs/2609.21983)
- DreamZero: over 2× generalisation improvement vs SOTA VLAs on real robots (AgiBot). Video-only cross-embodiment demos (10–20 min) give >42% relative improvement on unseen tasks — [arXiv](https://arxiv.org/abs/2602.15922); [project](https://dreamzero0.github.io/)
- Im2Flow2Act: 81% average over four real-world tasks (rigid, articulated, deformable); the robot is not named on the page — [project](https://im-flow-act.github.io)
- "Bridging Semantics and Kinematics" (training-free, Molmo2 router plus primitives): 62% end-to-end task success; arm not named — [alphaXiv](https://www.alphaxiv.org/abs/2606.23157)
- MolmoBot (not training-free; sim-generated data): 79.2% zero-shot real-world pick-and-place on a Franka FR3 (aggregator source) — [emergentmind](https://www.emergentmind.com/papers/2603.16861)
- GraspGen-X: NVIDIA reports out-of-the-box use for several common grippers. I did not extract quantitative success rates for unseen parallel grippers — [NVIDIA blog](https://blogs.nvidia.com/blog/cvpr-research-grasping-driving-agent-training/)

### Inferences
- For simple cube pick/place and stacking, a modular pipeline (segmentation → GraspGen-X grasp → IK/motion plan → VLM- or rule-chosen place pose) is likely more reliable than any of these cross-embodiment learned policies. Cube tasks are geometrically trivial, and failure modes come mostly from perception and calibration.

### Gaps
- No VoxPoser, ReKep, MOKA or CoPa success numbers were re-verified in this pass.
- No GraspGen-X success rate on unseen 2-finger grippers was extracted.

## Q5. Dependencies, compute, latency

### Takeaway
GraspGen-X is fast and local (milliseconds with TensorRT). ReKep and VoxPoser depend on OpenAI-class API LLMs. DreamZero needs a 14B model on at least 2 high-end GPUs and runs at about 0.6–3 s per call warm. KITE trains about 10 h on one consumer GPU per new embodiment.

### Cited Findings
- GraspGen-X: 3.178 ms for 150 candidates with the native C++ TensorRT runtime on an RTX 6000 Ada (synthetic object, excluding init/transfer) — [GraspGenX](https://github.com/NVlabs/GraspGenX)
- GraspGen: 20 Hz before TensorRT and 21× less memory than prior methods. It was tested on V100/A100/H100/L40s, and training used 8×A100 — [GraspGen](https://github.com/NVlabs/GraspGen)
- ReKep needs an `OPENAI_API_KEY` (GPT-4o) — [ReKep](https://github.com/huangwl18/ReKep)
- DreamZero needs Python 3.11 and CUDA 12.9+. It was tested on GB200 and H100, and distributed inference needs ≥2 GPUs. Warm inference is ~0.6 s on GB200 and ~3 s on H100, with TensorRT on GB200 only. The project page reports ~150 ms per chunk / 7 Hz with optimisations and a DreamZero-Flash single-step variant — [repo](https://github.com/dreamzero0/dreamzero); [project](https://dreamzero0.github.io/). These two latency claims differ; the 150 ms figure is presumably the optimised GB200 setting.
- KITE: about 10 h on one RTX 5090 to add the Wuji embodiment. The synthetic data is ~84 GB. MANO models are licensed separately — [project](https://kite-manip.github.io/); [repo](https://github.com/helloqxwang/KITE)

### Inferences
- A local VLM (e.g., Molmo, already served in this repo per the project notes) can replace GPT-4o in a planner pipeline, as the Molmo2-router paper does. This avoids API dependence, at some cost in reasoning quality.

### Gaps
- VoxPoser API model requirements and per-step latency were not re-verified.

## Q6. Maturity and Isaac Sim/Lab integration

### Takeaway
GraspGen and GraspGen-X are the most mature: NVIDIA-maintained, updated through mid-2026, with USD/Isaac Sim export and an Isaac Lab grasp-data generator. ReKep's only sim integration is OmniGibson on an old Isaac Sim 2023.1.1. KITE is a single-commit MuJoCo release. None of the family-1/3 repos showed Isaac Lab 3.0 examples.

### Cited Findings
- GraspGen changelog runs from 2025-07 to 2026-06 (collision filtering, ZMQ server, MCP tool-calling, MoE). GraspDataGen is an Isaac Lab-based data generator for pinch grippers (released 2025-09-30). The repo writes grasps into USD and can build an Isaac Sim USD with up to 10 environments for grasp testing — [GraspGen](https://github.com/NVlabs/GraspGen)
- GraspGen-X has no direct Isaac integration beyond USD export for Isaac Sim/Omniverse — [GraspGenX](https://github.com/NVlabs/GraspGenX)
- ReKep pins OmniGibson commit cc0316a on Isaac Sim 2023.1.1. Other robots are "currently untested" — [ReKep](https://github.com/huangwl18/ReKep)
- DreamZero support covers DROID sim eval and RoboArena; PolaRiS and Genie 3.0 sim support are marked "coming soon" — [repo](https://github.com/dreamzero0/dreamzero)
- KITE uses MuJoCo (`MUJOCO_GL=egl`) and has one commit — [repo](https://github.com/helloqxwang/KITE)

### Inferences
- The lowest-risk zero-shot path for SP7 in Isaac Lab 3.0 is to: render depth and segmentation from the fixed camera; run GraspGen-X with an SP7 `parallel_2f` gripper config generated from its URDF; filter grasps by IK reachability using the existing SP7 IK; pick the place pose by rule or with a local VLM (VoxPoser/ReKep-style constraint for "in box" or "on top of cube"); and execute through the existing waypoint expert. Every component is either released or already in the repo.
- That pipeline could also generate the about 30 minutes of SP7 play or success data that DreamZero-style fine-tuning would need, if a learned policy is wanted later.

### Gaps
- Licence compatibility of the NVIDIA Open Model License for the user's intended (possibly commercial) use was not assessed.
- Actual Isaac Lab 3.0 (Newton) compatibility of GraspGen-X's USD export is untested.
