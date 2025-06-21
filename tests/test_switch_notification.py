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
    token_limit = monitor.get_token_limit(plan)
    switched = False
    shown = False

    plan, token_limit, switched, shown, show = monitor.update_switch_state(
        8000, token_limit, plan, switched, shown, blocks
    )
    assert plan == "custom_max"
    assert token_limit == 10000
    assert switched is True
    assert show is True
    assert shown is True

    plan, token_limit, switched, shown, show = monitor.update_switch_state(
        9000, token_limit, plan, switched, shown, blocks
    )
    assert plan == "custom_max"
    assert token_limit == 10000
    assert switched is True
    assert show is False
    assert shown is True
