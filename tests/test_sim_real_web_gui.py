"""Tests for the paired sim/real browser GUI: command parsing and the live HTTP command queue."""

import importlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
gui_module = importlib.import_module("sim_to_real_so101.utils.sim_real_web_gui")
parse_command = gui_module.parse_command
SimRealWebGui = gui_module.SimRealWebGui

INFO = {
    "motion": False,
    "policy": "p",
    "checkpoint": "c",
    "instruction": "i",
    "eval_set": "e",
    "starts": [],
    "scene": {"table_outline_base_xy": [], "box_footprint_base_xy": []},
}


def _make_gui():
    return SimRealWebGui(INFO, port=0)


def _post(gui, body, raw=None):
    data = raw if raw is not None else json.dumps(body).encode()
    request = urllib.request.Request(
        gui.url + "command", data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status


# ── parse_command: valid bodies ──────────────────────────────────────────────


def test_parse_label_strips_note():
    assert parse_command({"cmd": "label", "stage": "grasped", "note": "  ok "}) == {
        "cmd": "label",
        "stage": "grasped",
        "note": "ok",
    }


def test_parse_label_without_note_gives_empty_note():
    assert parse_command({"cmd": "label", "stage": "placed"}) == {"cmd": "label", "stage": "placed", "note": ""}


@pytest.mark.parametrize("name", ["spawn", "placed", "run_sim", "run_real", "stop", "save", "discard", "exit"])
def test_parse_simple_commands(name):
    assert parse_command({"cmd": name}) == {"cmd": name}


def test_parse_spawn_ignores_extra_keys():
    assert parse_command({"cmd": "spawn", "x": 1}) == {"cmd": "spawn"}


# ── parse_command: invalid bodies ────────────────────────────────────────────


@pytest.mark.parametrize(
    "body",
    [
        pytest.param([], id="not_a_dict"),
        pytest.param({}, id="missing_cmd"),
        pytest.param({"cmd": "fly"}, id="unknown_cmd"),
        pytest.param({"cmd": "label", "stage": "flying"}, id="label_bad_stage"),
        pytest.param({"cmd": "label", "stage": "placed", "note": "a" * 501}, id="label_note_too_long"),
    ],
)
def test_parse_rejects_invalid(body):
    with pytest.raises(ValueError):
        parse_command(body)


# ── live server ──────────────────────────────────────────────────────────────


def test_queue_holds_one_request_and_pops_in_order():
    gui = _make_gui()
    try:
        assert _post(gui, {"cmd": "run_sim"}) == 204
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(gui, {"cmd": "placed"})
        assert excinfo.value.code == 409
        assert gui.pop_request() == {"cmd": "run_sim"}
        assert gui.pop_request() is None
    finally:
        gui.destroy()


def test_stop_is_flag_not_queued():
    gui = _make_gui()
    try:
        assert _post(gui, {"cmd": "run_sim"}) == 204
        assert _post(gui, {"cmd": "stop"}) == 204
        assert gui.stop_requested() is True
        assert gui.pop_request() == {"cmd": "run_sim"}
        assert gui.pop_request() is None
        gui.clear_stop()
        assert gui.stop_requested() is False
    finally:
        gui.destroy()


def test_busy_gui_rejects_commands_but_accepts_stop():
    gui = _make_gui()
    try:
        gui.set_status("x", busy=True)
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(gui, {"cmd": "run_sim"})
        assert excinfo.value.code == 409
        assert _post(gui, {"cmd": "stop"}) == 204
    finally:
        gui.destroy()


def test_bad_bodies_return_400():
    gui = _make_gui()
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(gui, None, raw=b"{")
        assert excinfo.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(gui, {"cmd": "fly"})
        assert excinfo.value.code == 400
    finally:
        gui.destroy()


def test_index_page_served():
    gui = _make_gui()
    try:
        html_path = ROOT / "source" / "gui" / "sim_real.html"
        if not html_path.exists():
            pytest.skip("source/gui/sim_real.html does not exist yet")
        with urllib.request.urlopen(gui.url, timeout=5) as response:
            assert response.status == 200
            assert b"sim_real.jsx" in response.read()
    finally:
        gui.destroy()


def test_unknown_path_returns_404():
    gui = _make_gui()
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(gui.url + "nope", timeout=5)
        assert excinfo.value.code == 404
    finally:
        gui.destroy()
