# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Sim keyboard requests for the single master teleoperation loop."""

import carb
import omni.appwindow


class KeyboardControl:
    """Collect keyboard requests without owning recorder state.

    The master loop consumes these flags once per control iteration. This keeps
    real and simulation episode transitions atomic from the application's point
    of view instead of dispatching independent recorder callbacks.
    """

    def __init__(self):
        self.recording = False
        self.start_episode_requested = False
        self.end_episode_requested = False
        self.rerecord_episode_requested = False
        self.stop_requested = False
        self.reset_world_requested = False

        self._window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._window.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return False

        key = event.input
        if key == carb.input.KeyboardInput.RIGHT:
            if self.recording:
                self.end_episode_requested = True
                print("[INFO]: Right Arrow: end synchronized episode requested.")
            return True

        if key == carb.input.KeyboardInput.LEFT:
            if self.recording:
                self.rerecord_episode_requested = True
                print("[INFO]: Left Arrow: re-record synchronized episode requested.")
            return True

        if key == carb.input.KeyboardInput.ESCAPE:
            self.stop_requested = True
            print("[INFO]: Escape: clean stop requested.")
            return True

        if key == carb.input.KeyboardInput.R:
            self.reset_world_requested = True
            if self.recording:
                self.end_episode_requested = True
            print("[INFO]: Reset requested.")
            return True

        if key == carb.input.KeyboardInput.S:
            if self.recording:
                self.end_episode_requested = True
                print("[INFO]: Stop/save synchronized episode requested.")
            else:
                self.start_episode_requested = True
                print("[INFO]: Start synchronized episode requested.")
            return True

        if key == carb.input.KeyboardInput.C:
            if self.recording:
                self.rerecord_episode_requested = True
                print("[INFO]: Cancel/re-record synchronized episode requested.")
            return True

        return False

    def consume_requests(self) -> dict[str, bool]:
        requests = {
            "start": self.start_episode_requested,
            "end": self.end_episode_requested,
            "rerecord": self.rerecord_episode_requested,
            "stop": self.stop_requested,
            "reset": self.reset_world_requested,
        }
        self.start_episode_requested = False
        self.end_episode_requested = False
        self.rerecord_episode_requested = False
        self.stop_requested = False
        self.reset_world_requested = False
        return requests

    def set_recording(self, recording: bool) -> None:
        self.recording = recording

    # Compatibility helpers for callers that previously toggled recording
    # through this object directly.
    def start_recording(self) -> None:
        if not self.recording:
            self.start_episode_requested = True

    def stop_recording(self) -> None:
        if self.recording:
            self.end_episode_requested = True

    def cancel_recording(self) -> None:
        if self.recording:
            self.rerecord_episode_requested = True

    @property
    def reset_world(self) -> bool:
        return self.reset_world_requested

    @reset_world.setter
    def reset_world(self, value: bool) -> None:
        self.reset_world_requested = value

    def cleanup(self) -> None:
        if self._sub_keyboard:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard, self._sub_keyboard
            )
            self._sub_keyboard = None
