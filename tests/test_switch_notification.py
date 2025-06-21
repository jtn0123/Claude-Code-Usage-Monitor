import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ccusage_monitor", Path(__file__).resolve().parents[1] / "ccusage_monitor.py"
)
assert spec and spec.loader
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_switch_notification_only_once():
    blocks = [{"isGap": False, "isActive": False, "totalTokens": 10000}]
    plan = "pro"
    state = monitor.MonitorState(token_limit=monitor.get_token_limit(plan))

    plan, show = monitor.update_switch_state(8000, plan, blocks, state)
    assert plan == "custom_max"
    assert state.token_limit == 10000
    assert state.switched_to_custom_max is True
    assert show is True
    assert state.switch_notification_shown is True

    plan, show = monitor.update_switch_state(9000, plan, blocks, state)
    assert plan == "custom_max"
    assert state.token_limit == 10000
    assert state.switched_to_custom_max is True
    assert show is False
    assert state.switch_notification_shown is True
