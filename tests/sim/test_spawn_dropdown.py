"""Both HTTP bridges preserve selected manifest names and reject unavailable objects."""
from dataclasses import replace
import json
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from core.rollout.paired_gui import PairedEvaluationGui
from core.tasks.task import load_task
from sim.recording.web_gui import RecordingWebGui
from sim.teleop.controls import CommandQueue
from utils.arms.arms import load_arm_spec


def post(url, body):
    return urlopen(Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}))


def multi_object_task():
    # Production manifests currently flag at most one spawnable; only test metadata changes.
    task = load_task("stack_bowls")
    objects = {name: replace(obj, params={**obj.params, "spawnable": True})
               for name, obj in task.spec.objects.items()}
    return SimpleNamespace(stages=task.stages, spec=replace(task.spec, objects=objects))


def test_recording_http_state_and_queue_preserve_the_second_object():
    names = [name for name, obj in multi_object_task().spec.objects.items() if obj.params.get("spawnable")]
    gui = RecordingWebGui([], port=0)
    try:
        gui.set_state(False, True, "Ready", spawnable=names)
        assert json.loads(gui._payload)["spawnable"] == names
        assert post(gui.url + "command", {"cmd": "spawn", "object": names[1]}).status == 204
        controls = CommandQueue(enable_spawn=True)
        controls.submit_gui(gui.consume_requests())
        assert controls.consume_requests() == {"spawn": True, "object": names[1]}
        gui.set_state(False, True, "Ready")  # Other state updates retain the task's choices.
        assert json.loads(gui._payload)["spawnable"] == names
        for body in ({"cmd": "spawn", "object": "unavailable_object"}, {"cmd": "spawn"}):
            with pytest.raises(HTTPError):
                post(gui.url + "command", body)
        assert not any(gui.consume_requests().values())
    finally:
        gui.destroy()


def test_paired_http_lists_manifest_objects_and_passes_the_selected_name(tmp_path, monkeypatch):
    task = multi_object_task()
    monkeypatch.setattr("core.rollout.paired_gui.load_task", lambda name: task)
    args = SimpleNamespace(arm_spec=load_arm_spec("so101")[0], task="stack_bowls", scene="rpd_meeting_room",
        follower_id="fixture", follower_port="fixture", camera_external=None, camera_gripper=None,
        rename_map="{}", backend="s2r", no_object=False, enable_motion=False, policy="fixture",
        policy_host="localhost", policy_port=5556, lang_instruction="Stack the objects", episode_seconds=1.,
        fps=30, gui_host="127.0.0.1", gui_port=0)
    gui = PairedEvaluationGui(args, [], tmp_path, manifests={})
    try:
        expected = [name for name, obj in task.spec.objects.items() if obj.params.get("spawnable")]
        assert len(expected) > 1
        assert gui.snapshot()["spawnable"] == gui.catalog()["spawnable"] == expected
        gui.publish("ready", "Ready", connected=True)
        with pytest.raises(HTTPError):
            post(gui.url + "/api/spawn", {"object": "unavailable_object"})
        assert not gui.consume_requests()
        assert post(gui.url + "/api/spawn", {"object": expected[1]}).status == 202
        assert gui.consume_requests() == {"spawn": expected[1]}
    finally:
        gui.destroy()
