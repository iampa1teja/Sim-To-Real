# Zero-shot open generalist policies on a new low-cost arm (SP7): evidence and integration requirements

Scope note: researched 2026-11-11 with about 17 search/fetch calls. Several key papers were only partly readable: the MolmoAct2 PDF was too large to fetch, and the HTML was truncated before its real-robot tables. Some figures come from secondary summaries and are marked as such. "Same robot, new scene" and "new robot" are kept separate throughout.

## Q1. Reported zero-shot / near-zero-shot results on embodiments outside training data (esp. low-cost arms)

### Takeaway
Released open VLAs transfer zero-shot well only to new scenes on an embodiment they were trained on (DROID-Franka, and SO-100/101 for MolmoAct2). On a genuinely new arm, success falls sharply. Recovering it took research-grade interventions: a source-arm joint interface with FK/IK retargeting, end-effector masking, and EEF-delta representations. The best real-world zero-shot new-arm figure found is about 70% (Cloak on a YAM arm), and it needed a custom retrained model. I found no credible report of an off-the-shelf checkpoint working zero-shot on a custom low-cost arm. One community evaluation of MolmoAct2 on unseen SO-100 rigs found it worse than a "hold current pose" baseline.

### Cited Findings
**Same robot, new scene (not the SP7 case)**
- openpi says π0-FAST-DROID "can perform a wide range of simple table-top manipulation tasks 0-shot in new scenes on the DROID robot platform", and π0-ALOHA-towel folds towels zero-shot on ALOHA. It also warns that the expert checkpoints "may or may not work on your particular robot", and that π0 was developed on PI's own robots, so adapting to other platforms is not expected to always succeed. — [openpi README](https://github.com/Physical-Intelligence/openpi)
- Penn PAL Lab "π0 in the wild" ran 300+ trials of π0-FAST-DROID on a Franka/Robotiq DROID-style setup. Average task progress was 42.3%, with roughly 19-28% binary success by category (pick-and-place: 38.4% progress, 24% success). Unseen background and OOD objects scored 0%. The authors call it a "vibe-check" evaluation, not a benchmark. — [Penn PAL Lab](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild)
- MolmoAct2: the DROID and SO-100/101 checkpoints are described as deployable out of the box. The paper's limitations appendix lists "Embodiment-specific zero-shot deployment", meaning zero-shot is tied to its large-data embodiments. — [MolmoAct2 arXiv 2605.02881](https://arxiv.org/html/2605.02881)
- A secondary summary claims MolmoAct2 reaches 87.1% zero-shot on real-world DROID tasks. This could not be verified in the primary text read. — [Emergent Mind summary](https://www.emergentmind.com/papers/2605.02881)

**New robot (the SP7 case)**
- Cloak (Stanford, arXiv 2606.22836, June 2026) tested π0.5 fine-tuned on DROID (Franka + Robotiq) zero-shot on unseen end effectors and arms. The baseline π0.5-droid with tip-pose retargeting scored 77.1% on the source, 37.5% on the UMI gripper, 43.8% on the YAM arm and 16.7% on the Sharpa hand. Cloak-VLA (wrist-cam EE masking) scored 75.0 / 68.8 / 70.8 / 62.5%. LAP-VLA (language-action alignment, 6-DoF wrist pose output, custom IK) scored 75.0 / 31.2 / 29.2 / 31.2%. Task-progress numbers: π0.5-droid 89.3 / 69.0 / 69.5 / 54.4, Cloak 88.0 / 85.1 / 86.3 / 81.8. — [Cloak arXiv](https://arxiv.org/html/2606.22836v1), [project page](https://tml.stanford.edu/cloak/)
- In Cloak, the action/state interface stayed in the source's 8-D joint space. Model joints → FK → source jaw-tip poses → IK (Mink) → target joints, and the reverse for the state input. The model's gripper signal overrides IK for the jaws. — [Cloak arXiv](https://arxiv.org/html/2606.22836v1)
- ZETA (Galbot/PKU, arXiv 2609.02546, Sep 2026) is a controlled study with a π0.5-style backbone pretrained on 512 procedural Franka-family embodiments. Average simulated zero-shot task progress on held-out targets was 75.7% in the best configuration, 74.3% for arm-only shifts (UR5e, GoogleRobot) and 60.4% for full-embodiment shifts. Replacing 5% of pretraining with target-embodiment data raised UR5e+UMI from 69.3 to 78.6 and GoogleRobot from 51.4 to 68.9, still below the target-specific oracle. Source diversity (512 vs 1 source) added about 18 points at a fixed 640K-trajectory budget. — [ZETA arXiv](https://arxiv.org/html/2609.02546v1)
- Being-H0.5 reports a "first empirical instance of embodiment-level zero-shot task completion (albeit at low success rates)" on a new robot. — [Being-H0.5 arXiv 2601.12993](https://arxiv.org/pdf/2601.12993)
- Community, single unreviewed author (ANECDOTAL): an offline evaluation of the MolmoAct2 SO-100/101 checkpoint on 25,800 frames from community SO-100 rigs. On frames from rigs it had not trained on (about 69% of the panel), action-prediction error was about twice the "state-copy" (hold pose) baseline. The author says the model "predicts kinematically sane trajectories in the wrong part of the workspace" and identifies visual workspace calibration on unseen rigs as the binding constraint. The extracted summary also listed pooled/unseen/seen errors (13.87 / 7.00 / 16.97 vs state-copy 8.32) that seem to contradict this, so verify against the page. — [blog: MolmoAct2 OOB results](https://mcobzarenco-fontaine-blog.static.hf.space/posts/2026-08-10-molmoact2-oob-results.html)
- Community, ANECDOTAL: an NVIDIA forum user reported 0% success recreating a tutorial's SO-101 real-robot eval, then about 80% over 10 episodes after changing the setup. It is unclear which policy was used. — [NVIDIA forum](https://forums.developer.nvidia.com/t/physical-modelling-of-sim2real-so101-arm-project/371627)
- GR00T on SO-100/101: community model cards for N1.5/N1.6/N1.7 SO-101 fine-tunes mostly report loss or single episodes and explicitly claim no rollout success rate. One LeIsaac sim checkpoint reports 3/3 oranges placed in one episode. — [GR00T N1.7 SO101 card](https://huggingface.co/robocurve/gr00t-n1.7-so101-molmoact2), [GR00T N1.6 PickOrange](https://huggingface.co/wsagi/GR00T-N1.6-PickOrange), [Seeed N1.6 guide](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.6_for_lerobot_so_arm_and_deploy_on_agx_orin/)

### Inferences
- For SP7 (7-DoF STS servos, prismatic two-finger gripper, custom cameras), the realistic expectation for any released checkpoint run directly is near 0% to low success. The only demonstrated path to roughly 40-70% zero-shot is to present the policy with a known source embodiment. That means retargeting to a DROID-Franka (or SO-101) joint/EEF interface through FK/IK, ideally with gripper masking in the wrist view. Without retraining, Cloak's no-mask baseline (π0.5-droid + retargeting: 37-44% on new arm/gripper) is the most comparable number.
- SP7's 7 revolute joints make Franka/DROID the natural retarget source: tip-pose IK to a 7-DoF arm is well-posed. SO-101 checkpoints expect a 5-DoF + gripper absolute joint space that SP7 cannot reproduce joint-for-joint.
- Workspace scale matters. A low-cost arm's reach and its camera-to-table geometry differ from Franka's, so pose retargeting has to rescale or crop the workspace. This is likely the same "wrong part of the workspace" failure seen in the SO-100 community evaluation.

### Gaps
- No primary-source zero-shot numbers were found for SO-100/101, Koch, WidowX, xArm or UR5 running an off-the-shelf generalist without fine-tuning. The MolmoAct2 SO-100 table (Appendix C.4) and its comparisons with DePi/SmolVLA were not readable.
- No LeRobot Discord or GitHub issue threads were retrieved; Discord is not searchable.
- RoboArena (distributed DROID evals) was not covered.

## Q2. Simulation benchmarks testing cross-embodiment or visual sim-to-real gaps

### Takeaway
Simulation evaluation of open VLAs is mature for their own embodiments: DROID in Isaac Lab Arena, MolmoSpace, SimplerEnv for Google Robot/WidowX, and LIBERO. Sim results correlate with real only when visuals are matched. Cross-embodiment zero-shot sim results exist mainly in controlled research (ZETA). Isaac Lab Arena already ships an openpi client with a pluggable embodiment adapter, which is the closest template for an SP7 sim harness.

### Cited Findings
- SimplerEnv "visual matching" (real background overlay plus texture matching) vs "variant aggregation" (averaging over backgrounds, lighting, distractors, textures). A secondary summary reports Google-robot visual-matching Pearson r 0.924 and MMRV 0.056, vs 0.308 / 0.375 for a validation-loss baseline (verify against the paper). — [SimplerEnv repo](https://github.com/simpler-env/SimplerEnv), [paper arXiv 2405.05941](https://arxiv.org/pdf/2405.05941), [secondary wiki](https://aiwiki.ai/wiki/simpler_env)
- Isaac Lab Arena + openpi: OpenPI runs as a websocket server (`serve_policy.py`, port 8000). Arena's `Pi0RemotePolicy` client uses embodiment `droid_abs_joint_pos` with adapter `droid`, and checkpoints `pi05_droid_jointpos_polaris` (open-loop horizon 15) or `pi0_droid_jointpos_polaris` (horizon 10). In a 9-run × 3-episode multi-variation test (object, HDR background, destination), 8/9 runs reached 1.0 success and sugar_box reached 0.0. The sample is small, and it is the same embodiment. A new embodiment needs a subclass of `Pi0EmbodimentAdapter` registered in `_resolve_openpi_embodiment_adapter`. A separate GR00T guide exists. — [Isaac Lab Arena openpi docs 0.3.1](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.1/pages/quickstart/running_a_real_policy/openpi.html)
- MolmoSpace (sim, DROID variants): MolmoAct2-DROID averages 37.7% vs π0.5-DROID 34.5% vs π0-DROID 23.2%. Per task: Pick 43.7 vs 36.4, Pick & Place 26.7 vs 13.6, Close 70.8 vs 65.1, Open 9.5 vs 22.7. — [MolmoAct2 arXiv](https://arxiv.org/html/2605.02881)
- ZETA (sim, held-out embodiments): 75.7% average progress in the best configuration. In real-world validation (2 tasks), the best configuration averaged 89.9%, vs 56.0% for absolute-EEF / world-delta. — [ZETA](https://arxiv.org/html/2609.02546v1)
- GR00T N1 → N1.5, sim: RoboCasa at 30 demos/task 17.4 → 47.5%; Sim GR-1 at 30 demos/task 43.2 → 47.4%; Sim GR-1 zero-shot 39.6 → 43.9%. — [NVIDIA GR00T N1.5 page](https://research.nvidia.com/labs/gear/gr00t-n1_5/)
- LIBERO, a same-embodiment fine-tuned sim benchmark: OpenVLA-OFT averages 97.1% across four suites ([OpenVLA-OFT](https://arxiv.org/html/2502.19645v2)). SmolVLA (0.45B) scores 87.3% vs π0 86.0% (secondary summaries of [SmolVLA arXiv 2506.01844](https://www.emergentmind.com/papers/2506.01844)).

### Inferences
- LIBERO/RoboCasa numbers measure fine-tuned, in-distribution performance and say nothing about SP7 zero-shot. The relevant sim signal is a DROID-style adapter test in the project's own Isaac Lab scene.
- The project's sim scenes will not match the visual statistics of DROID or SO-100 data. SimplerEnv's evidence implies that sim success without visual matching is a weak predictor of real success.

### Gaps
- No public sim benchmark evaluates zero-shot transfer of a released checkpoint to a low-cost arm.
- RoboCasa and RoboArena cross-embodiment numbers were not retrieved. Isaac Lab Mimic → GR00T sim eval numbers beyond NVIDIA's own pages were not found.
- SimplerEnv variant-aggregation correlation figures were not retrieved.

## Q3. Sensitivity to viewpoint, appearance, control frequency, gripper; known tricks

### Takeaway
The biggest sensitivities are the wrist camera and seeing an unfamiliar gripper or arm in it, the state/action representation (absolute vs EEF-relative), camera pose relative to the training distribution, and prompt wording. Documented tricks are retargeting through the source embodiment's kinematics, masking the end effector in the wrist view, EEF-delta state/action, aligning camera poses so the gripper's image location matches training, async chunking with overlap aggregation, and running at the checkpoint's native chunk length and rate.

### Cited Findings
- Wrist-cam appearance of the end effector is the dominant cross-embodiment failure. Masking it (with augmentation) raised the Sharpa hand from 16.7% to 62.5% and the YAM arm from 43.8% to 70.8%. Without mask augmentation, Sharpa scored 39.6%. — [Cloak](https://arxiv.org/html/2606.22836v1)
- State representation: switching from absolute EEF state to EEF-delta raised arm-only transfer from about 38.5-39.9% to 69.8-74.3%, and full-embodiment transfer from about 33.9-38.6% to 58.4-60.4% (sim). EEF-centred representations add about 15 points on average. — [ZETA](https://arxiv.org/html/2609.02546v1)
- Camera pose: ZETA aligned the wrist and third-person cameras "so the gripper's image-plane distribution stays close to the post-training distribution". It ran no viewpoint ablation. — [ZETA](https://arxiv.org/html/2609.02546v1)
- Occlusion and prompts with π0-FAST-DROID: blocking the wrist camera gave 0% (robot froze), blocking the side camera mid-trial gave 50%. "Close the toilet" scored 0% vs 100% for "close the white lid of the toilet". Images were 224×224. Sample sizes were small (4-12 trials). — [Penn PAL Lab](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild)
- Native action interfaces in MolmoAct2: DROID uses absolute joint pose, 15-step chunk at 15 Hz. SO-100/101 and YAM use absolute joint pose, 30 steps at 30 Hz. LIBERO uses relative EEF, 10 steps at 10 Hz. The flow expert pads to 30 steps × 32 dims. — [MolmoAct2](https://arxiv.org/html/2605.02881)
- Control rate and latency: autoregressive VLAs run at 3-5 Hz, too slow for 25-50+ Hz control. OFT-style parallel decoding with 25-step chunks gives 43× throughput, and ALOHA ran at 25 Hz. — [OpenVLA-OFT](https://arxiv.org/html/2502.19645v2)
- Async chunking: LeRobot async inference uses `actions_per_chunk` (default 50, typically 10-50), `chunk_size_threshold` (the table says default 0.7, but the examples use 0.5 and the text says "0.5-0.6 work well"), and `aggregate_fn_name=weighted_average` for overlapping chunks. Lower the fps if the queue runs dry. π0 needs about 14 GB at inference and SmolVLA about 2 GB. — [LeRobot async docs](https://huggingface.co/docs/lerobot/main/en/async)
- SmolVLA async: comparable success, about 30% faster completion (9.7 s vs 13.75 s), and roughly 2× more cycles in a fixed 60 s window (19 vs 9). Figures are from secondary summaries. — [SmolVLA summaries](https://www.emergentmind.com/papers/2506.01844)
- Gripper: ZETA's gripper-only shift (Franka arm + UMI gripper) scored 78.3% in the best configuration. In Cloak, the model's gripper scalar is mapped directly to the target jaw opening rather than solved through IK. — [ZETA](https://arxiv.org/html/2609.02546v1), [Cloak](https://arxiv.org/html/2606.22836v1)

### Inferences
- SP7's prismatic two-finger gripper resembles a Robotiq/Franka parallel jaw, so a scalar gripper mapping (model open/close → SP7 stroke in metres) is low-risk compared with the visual appearance mismatch in the wrist camera.
- Absolute-joint checkpoints (DROID jointpos, SO-101) are the worst fit for a new arm because their action space is the joint space of the training robot. A retarget layer is mandatory: source-joints → FK → TCP pose → SP7 IK. The project's existing differential IK controller and arm-declared `ik` TCP constraints are the natural place for it.
- Hobby STS servos have backlash and lower bandwidth, so commanded chunks at 15-30 Hz should be smoothed (overlap weighted-average) and possibly rate-limited.

### Gaps
- No quantitative ablation of camera viewpoint shift for EE-action VLAs was retrieved; that evidence is qualitative or anecdotal.
- No quantified sim-render-vs-real appearance gap was found for these specific models.
- No published "delta scaling" numbers were found.

## Q4. Minimum adaptation data when zero-shot fails

### Takeaway
Most published adaptations use tens to hundreds of demos per task (ZETA real post-training: 50/task; GR00T sim at 30/task), plus tens of hours for a new platform. Exact numbers for a brand-new low-cost arm from GR00T/π0/SmolVLA/MolmoAct2/OpenVLA-OFT were not retrievable in this pass. The most actionable finding is ZETA's: a small target-embodiment share (5%) mixed into training gave a 13-point gain. LoRA fine-tuning of π0 fits in about 22.5 GB VRAM.

### Cited Findings
- ZETA real-world post-training used 50 demonstrations per task. Sim post-training used 40K trajectories per task. Swapping 5% of pretraining data for target data added +9.3 / +17.5 points. — [ZETA](https://arxiv.org/html/2609.02546v1)
- GR00T N1.5 reaches 47.5% on RoboCasa with 30 demos per task (N1: 17.4%). GR00T N1 claims that with only 10% of the human teleop data it beats Diffusion Policy trained on the full data by 32.4% (secondary summary). Unitree G1 used 1K teleop episodes: N1 44.0% → N1.5 98.8%. — [GR00T N1.5 page](https://research.nvidia.com/labs/gear/gr00t-n1_5/), [GR00T N1 arXiv summary results](https://arxiv.org/html/2503.14734v2)
- SmolVLA on SO-100 real tasks (pick-place, stacking, sorting) averaged 78.3% vs π0 61.7% vs ACT 48.3%. Community pretraining raised SmolVLA from 51.7% to 78.3%. Secondary summaries; per-task demo counts were not retrieved. — [SmolVLA summaries](https://www.emergentmind.com/papers/2506.01844)
- MolmoAct2-BimanualYAM: a new-platform checkpoint built from 34.5k demos (720+ hours, 28+ tasks). This is the scale used to add a supported zero-shot embodiment, not a minimal adaptation. — [MolmoAct2](https://arxiv.org/html/2605.02881)
- OpenVLA-OFT on ALOHA beat π0, RDT-1B, Diffusion Policy and ACT by up to 15% absolute average success. Demo counts were not retrieved. — [OpenVLA-OFT](https://arxiv.org/html/2502.19645v2)
- Compute: openpi needs more than 8 GB for inference, more than 22.5 GB for LoRA fine-tuning and more than 70 GB for full fine-tuning (JAX path; the PyTorch path lacks LoRA). The Seeed guide says GR00T N1.6 fine-tuning needs more than 48 GB VRAM. — [openpi](https://github.com/Physical-Intelligence/openpi), [Seeed](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.6_for_lerobot_so_arm_and_deploy_on_agx_orin/)

### Inferences
- For SP7, the cheapest credible path once zero-shot fails is 50-100 sim-generated demos per task (the project's scripted expert/Mimic pipeline already produces these), followed by a LoRA or GR00T post-train with a `new_embodiment` head. A small real-demo set (around 20-50) then addresses the visual gap. The project already has schema-2 GR00T preparation for SO-101, so tooling cost is low. Expected success is unknown until measured.

### Gaps
- Primary per-task demo counts for SmolVLA's SO-100 tasks, the OpenVLA-OFT ALOHA tasks, MolmoAct2's YAM fine-tune and π0's fine-tuning eval were not retrieved.
- No published data-scaling curve (10/25/50 demos) for a new low-cost arm was found.

## Q5. Integration effort: adapters, servers, LeRobot/Isaac Lab support

### Takeaway
Serving infrastructure is a solved problem: the openpi websocket server, LeRobot async gRPC PolicyServer/RobotClient, and the project's existing ZMQ clients. The real integration cost for SP7 is an embodiment adapter. It must (a) translate SP7 state into the checkpoint's expected proprio (source-arm joints via IK, or EEF pose), (b) translate output chunks back to SP7 absolute joint targets via FK→IK, (c) map the gripper scalar to prismatic stroke, and (d) present cameras matching the checkpoint's keys, resolution and approximate pose. Isaac Lab Arena's `Pi0EmbodimentAdapter` is a direct template for the sim side.

### Cited Findings
- openpi: a websocket policy server (`scripts/serve_policy.py`, port 8000) streams actions to a remote robot client, with a minimal client in `docs/remote_inference.md`. — [openpi](https://github.com/Physical-Intelligence/openpi)
- Isaac Lab Arena: `Pi0RemotePolicy` websocket client plus a `Pi0EmbodimentAdapter` subclass per embodiment. The DROID adapter uses absolute joint positions. — [Arena openpi docs](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.1/pages/quickstart/running_a_real_policy/openpi.html)
- LeRobot async inference: a `PolicyServer` populated at handshake, and a `RobotClient` wrapping a LeRobot `Robot` that streams observations and receives chunks over gRPC. It works with "all LeRobot-supported policies" (ACT, SmolVLA, π0 etc.). Camera keys must match the policy's config. — [LeRobot async docs](https://huggingface.co/docs/lerobot/main/en/async)
- Cloak-style retargeting used the Mink IK library with two designated tip points (the jaws for a gripper). — [Cloak](https://arxiv.org/html/2606.22836v1)
- LAP-VLA outputs a 6-DoF wrist pose, which requires the user's own IK on the target arm (Cloak authors did this for YAM). — [Cloak](https://arxiv.org/html/2606.22836v1)

### Inferences
- Estimated adapter work for SP7 (engineering judgement, not sourced):
  1. DROID/Franka kinematic model plus FK in the adapter.
  2. Workspace scaling/offset from Franka frame to SP7 base frame, using the XYZW/WXYZ boundary conventions the project already has.
  3. Reuse the SP7 differential IK.
  4. Gripper scalar → metres.
  5. Camera key/resolution remap (224×224 for π0 family).
  6. Optional wrist-cam gripper mask.

  Items 1-5 fit the existing `configure(ArmInfo)` client contract if the model-specific conventions stay inside the policy folder, as CLAUDE.md requires.
- The current ZMQ clients output absolute joint targets in the policy's own joint space, so they cannot drive SP7 directly from DROID/SO-101 checkpoints without this retarget layer.

### Gaps
- Not verified: current LeRobot (v0.6.x/v0.7) support for EE-action policies or a built-in EE→joint processor. The v0.7 async page was a redirect, and no EE-specific docs were fetched.
- An OpenVLA REST deploy server exists in the OpenVLA repo, but it was not fetched here.
- No Isaac Lab-native VLA runner beyond Arena's openpi/GR00T clients was confirmed.
