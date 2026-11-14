"""Robot-agnostic driver for a served policy.

Owns the action queue, optional smoothing (prefetch / blending / EMA / temporal ensembling), the optional
grasp controller, per-episode seeding and timing. The policy plugin (policies/<name>/client.py) decides
what the request contains and how the reply is read. Everything here works in raw LeRobot units keyed by
``<joint>.pos`` (-100..100 body, 0..100 gripper), so the real follower uses it directly and the simulator
wraps it (sim.eval.sim_remote_policy) to convert radians.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime

import numpy as np

from .adapter import build_plugin_client, check_server
from .contract import ArmInfo
from .discovery import get_manifest
from .smoothing import SimulationSmoother, SmoothingConfig, TimingLog



class RemotePolicy:
    """Next raw action for any robot from a policy server; one instance per rollout."""

    def __init__(
        self,
        joint_keys,
        policy_name: str = "groot_n17",
        host: str = "localhost",
        port: int | None = None,
        action_horizon: int | None = None,
        lang_instruction: str = "",
        prefetch_steps: int = 0,
        blend_steps: int = 0,
        ema_alpha: float = 1.0,
        smooth_gripper: bool = False,
        temporal_ensemble: int = 0,
        te_decay: float = 0.01,
        log_timing: bool = False,
        timing_csv: str = "",
        grasp_config=None,
        seed_per_episode: bool = False,
        episode_seed: int = 1984,
        control_dt: float = 1 / 30,
        policy_options: dict | None = None,
        api_token: str | None = None,
        timeout_ms: int = 15000,
        action_repeat: int = 1,
        arm_info: ArmInfo | None = None,
        async_inference: bool = False,
        refill_seconds: float = 0.5,
        starvation_grace: float = 1.0,
        clock=None,
    ):
        if not isinstance(action_repeat, int) or isinstance(action_repeat, bool) or action_repeat < 1:
            raise ValueError("action_repeat must be a positive integer")
        self.joint_keys = list(joint_keys)
        self.arm_info = arm_info
        if arm_info is not None and tuple(self.joint_keys) != arm_info.feature_names:
            raise ValueError('Remote policy joint order differs from ArmInfo')
        action_horizon = get_manifest(policy_name).default_horizon if action_horizon is None else action_horizon
        if type(action_horizon) is not int or action_horizon < 1:
            raise ValueError('action_horizon must be a positive integer')
        gripper_indices = arm_info.gripper_indices if arm_info is not None else ()
        self.grasp = None
        if grasp_config is not None and grasp_config.enabled:
            from .grasp import GraspController

            self.grasp = GraspController(grasp_config, self.joint_keys, control_dt, arm_info=arm_info)
            self.grasp.episode = -1  # the first reset starts episode zero
        self._seed_per_episode = seed_per_episode
        self._episode_seed = episode_seed
        self._host = host
        self._port = port
        self._policy_name = policy_name
        self._policy_options = dict(policy_options or {})
        self._api_token = api_token
        self._timeout_ms = timeout_ms
        self._action_horizon = action_horizon
        self._action_repeat = action_repeat
        self._effective_action_horizon = action_horizon * action_repeat
        self._lang_instruction = lang_instruction
        self.async_inference = async_inference
        self._action_queue: deque = deque()
        self._client = None
        self.server_info = None
        self.policy_cameras = None
        self.joint_transform = None
        self.inference_calls = 0
        options = SmoothingConfig(prefetch_steps, blend_steps, ema_alpha, smooth_gripper,
                                  temporal_ensemble, te_decay, log_timing)
        options.validate(self._effective_action_horizon)
        self._timing = None
        if log_timing:
            path = timing_csv or f"outputs/timing/policy_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
            self._timing = TimingLog(path, self.joint_keys, gripper_indices=gripper_indices)
        self._smoothing = (SimulationSmoother(options, self._effective_action_horizon, self.joint_keys,
                                              self._timing, grasp=self.grasp, gripper_indices=gripper_indices)
                           if options.enabled and not async_inference else None)
        self._async = None
        if async_inference:
            from .async_worker import AsyncInference
            self._async = AsyncInference(
                self._build_client, self._initialize_client,
                lambda actions: self._decode_action_chunk(actions, full_chunk=True),
                self.joint_keys, self._effective_action_horizon, control_dt, options,
                timeout=timeout_ms / 1000, refill_seconds=refill_seconds,
                starvation_grace=starvation_grace, clock=clock, gripper_indices=gripper_indices,
                grasp=self.grasp, timing=self._timing, seed_per_episode=seed_per_episode,
                episode_seed=episode_seed)

    @property
    def lang_instruction(self):
        return self._lang_instruction

    @lang_instruction.setter
    def lang_instruction(self, text):
        self.set_instruction(text)

    def set_instruction(self, text):
        """Invalidate old-task playback and in-flight replies before changing intent."""
        if not isinstance(text, str):
            raise TypeError('instruction must be a string')
        if self._async is not None:
            with self._async.lock:
                self._lang_instruction = text
                self._async.instruction_changed()
        else:
            self._lang_instruction = text

    @property
    def last_tick(self):
        return self._async.last_tick if self._async is not None else None

    def cancel(self):
        """Stop async playback immediately; reset is required to resume."""
        if self._async is not None:
            self._async.cancel()

    def record_applied_target(self, action):
        if self._async is not None:
            self._async.applied(action)

    @property
    def client(self):
        if self._async is not None:
            raise RuntimeError('The async policy client belongs exclusively to its worker thread')
        return self._client

    def _build_client(self):
        return build_plugin_client(
            self._policy_name, self.joint_keys, host=self._host, port=self._port,
            timeout_ms=self._timeout_ms, api_token=self._api_token, policy_options=self._policy_options,
            arm_info=self.arm_info)

    @staticmethod
    def _initialize_client(client):
        return dict(server_info=check_server(client),
                    cameras=list(getattr(client.policy, 'cameras', client.manifest.cameras)),
                    joint_transform=getattr(client.policy, 'joint_transform', None))

    def connect(self):
        """Connect outside the control loop; async sockets never leave their owner."""
        if self._async is not None:
            metadata = self._async.connect()
            self.policy_cameras = metadata['cameras']
            self.joint_transform = metadata['joint_transform']
            self.server_info = metadata['server_info']
            print(f'[INFO]: Async policy server connected; camera inputs: {self.policy_cameras}', flush=True)
            return
        self._client = self._build_client()
        cameras = getattr(self._client.policy, 'cameras', self._client.manifest.cameras)
        self.policy_cameras = list(cameras)
        print(f"[INFO]: Policy camera inputs: {self.policy_cameras}", flush=True)
        self.joint_transform = getattr(self._client.policy, "joint_transform", None)
        if self.joint_transform is not None:
            print(f"[INFO]: Policy joint transform: {self.joint_transform}")
        print(f"[INFO]: Connecting to the {self._policy_name} policy server at "
              f"{self._host}:{self._client.client.port}...")
        self.server_info = check_server(self._client)
        print("[INFO]: Policy server connected")

    def reset(self, seed=None):
        """Reset state; a benchmark seed must be acknowledged by the server."""
        if self._async is not None:
            self._async.reset(seed)
            self.inference_calls = 0
            return
        self._client.reset()
        if self.grasp is not None:
            self.grasp.reset()
        if seed is not None:
            self._episode_seed = int(seed)
        self._action_queue.clear()
        self.inference_calls = 0
        if self._smoothing is not None:
            self._smoothing.reset()
        if seed is not None:
            response = self._client.call_endpoint("seed_policy", {"seed": int(seed)})
            if response != {"seed": int(seed)}:
                raise RuntimeError("Policy seed not acknowledged; the server must implement seed_policy")

    @property
    def steps_until_query(self):
        """Control steps until the next observation is queried (for rendering)."""
        if self._async is not None:
            return 0  # Async consumers need current observations, including during refill.
        return (self._smoothing.steps_until_query if self._smoothing is not None
                else len(self._action_queue))

    def close_timing(self):
        if self.grasp is not None:
            self.grasp.close()
        if self._timing is not None:
            self._timing.close()

    def close(self):
        """Close a session, including its socket, on the policy worker thread."""
        if self._async is not None:
            self._async.close()
        self.close_timing()
        if self._client is not None:
            self._client.client.close()
            self._client = None

    def _decode_action_chunk(self, action_chunk, full_chunk=False):
        """(T, J) targets -> one {joint key: value} per control tick, each target repeated action_repeat times.

        full_chunk keeps the model tail for stale-prefix removal, blending and temporal overlap; the smoother
        caps execution separately.
        """
        action_chunk = np.asarray(action_chunk)
        horizon = action_chunk.shape[0] if full_chunk else min(action_chunk.shape[0], self._action_horizon)
        return [
            {name: float(action_chunk[t][i]) for i, name in enumerate(self.joint_keys)}
            for t in range(horizon)
            for _ in range(self._action_repeat)
        ]

    def _query(self, make_input, full_chunk):
        model_input = make_input()
        if self._seed_per_episode:
            from .grasp import seed_query

            seed_query(self._client, self._episode_seed)
        action_chunk = self._client.get_action(model_input)
        self.inference_calls += 1
        return self._decode_action_chunk(action_chunk, full_chunk=full_chunk)

    def next_raw_action(self, make_input, measured_gripper=None) -> dict:
        """Next action as {joint key: raw value}: queue, smoothing, grasp control and seeding.

        make_input() builds {"state", "images", "instruction"} and is called only when the policy is actually
        queried, so callers can defer image conversion. measured_gripper (raw units) is required when grasp
        control is enabled.
        """
        if self._async is not None:
            action = self._async.tick(make_input, measured_gripper)
            self.inference_calls = self._async.calls
            return action
        if self.grasp is not None:
            self.grasp.measured = float(measured_gripper)
        if self._smoothing is not None:
            return self._smoothing.get_action(lambda: self._query(make_input, True))
        if not self._action_queue:
            decoded = self._query(make_input, self.grasp is not None)
            if self.grasp is not None:
                self.grasp.observe_chunk(decoded, self.inference_calls - 1)
            self._action_queue.extend(decoded[:self._effective_action_horizon])
        action = self._action_queue.popleft()
        if self.grasp is not None:
            raw = np.asarray([action[k] for k in self.joint_keys])
            sent = self.grasp.process(raw, raw, self.inference_calls - 1)
            action = dict(zip(self.joint_keys, sent))
        return action

    def get_raw_action(self, state, images) -> dict:
        """Next action for a robot that reports raw LeRobot units (the real follower).

        state: current joints in joint_keys order, raw units. images: policy camera name -> HWC uint8 array.
        Returns {joint key: raw value}, ready for the follower's send_action.
        """
        state = np.asarray(state, dtype=np.float32)
        measured = float(state[self.grasp.gripper_index]) if self.grasp is not None else None
        return self.next_raw_action(
            lambda: {"state": state, "images": images, "instruction": self.lang_instruction}, measured)
