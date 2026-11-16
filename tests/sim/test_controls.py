"""GUI and terminal requests share FIFO transitions without keyboard/Kit imports."""
import pytest

from sim.teleop.controls import CommandQueue
from sim.teleop.config import AgentConfig, GuiConfig


def test_gui_and_terminal_commands_keep_order_and_resolve_toggle_at_consumption():
    controls = CommandQueue()
    controls.submit_text("start")
    controls.submit_gui({"save": True})
    controls.submit_text("s")
    assert controls.consume_requests() == {"start": True}
    controls.set_recording(True)
    assert controls.consume_requests() == {"end": True}
    controls.set_recording(False)
    assert controls.consume_requests() == {"start": True}
    assert controls.consume_requests() == {}


def test_reset_saves_active_recording_and_gui_rerecord_payload_stays_queued():
    controls = CommandQueue()
    controls.set_recording(True)
    controls.submit_text("reset")
    controls.submit_gui({"rerecord": True, "rerecord_episode": 0})
    assert controls.consume_requests() == {"reset": True, "end": True}
    assert controls.consume_requests() == {"prepare_rerecord": True, "rerecord_episode": 0}


def test_spawn_aliases_and_closed_queue():
    controls = CommandQueue(enable_spawn=True)
    controls.submit_text("s")
    assert controls.consume_requests() == {"spawn": True}
    assert not controls.submit_text("unknown")
    controls.cleanup()
    controls.submit("start")
    assert not controls.consume_requests()


@pytest.mark.parametrize("stop_on_eof", [False, True])
def test_terminal_eof_does_not_stop_a_gui_or_automated_eval(monkeypatch, stop_on_eof):
    def eof(*args):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    controls = CommandQueue()
    controls.start_terminal(stop_on_eof=stop_on_eof)
    controls._thread.join(1)
    assert controls.consume_requests() == ({"stop": True} if stop_on_eof else {})
    controls.cleanup()


def test_gui_only_requires_gui_and_keyboard_mode_is_removed():
    with pytest.raises(ValueError, match="terminal or gui"):
        AgentConfig(control_input="keyboard")
    with pytest.raises(ValueError, match="gui.enabled"):
        AgentConfig(control_input="gui")
    assert AgentConfig(control_input="gui", gui=GuiConfig(enabled=True)).control_input == "gui"
