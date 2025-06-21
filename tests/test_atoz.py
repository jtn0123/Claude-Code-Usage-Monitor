import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ccusage_monitor", Path(__file__).resolve().parents[1] / "ccusage_monitor.py"
)
assert spec and spec.loader
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_format_model_name_a_to_z():
    models = [
        ("claude-opus-4", "Opus"),
        ("claude-sonnet-4", "Sonnet"),
        ("claude-unknown", "claude-unknown"),
        (None, None),
    ]
    for model, expected in models:
        assert monitor.format_model_name(model) == expected
