#!/usr/bin/env python3

import subprocess
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import argparse
import urllib.request
from urllib.error import URLError
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from rich.console import Console, Group
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    RICH_AVAILABLE = False

# ANSI color codes for plain output
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"
COLOR_CYAN = "\033[96m"
COLOR_BLUE = "\033[94m"
COLOR_YELLOW = "\033[93m"
COLOR_WHITE = "\033[97m"
COLOR_GRAY = "\033[90m"


@dataclass
class MonitorState:
    """Tracking state for plan switching notifications."""

    token_limit: int
    switched_to_custom_max: bool = False
    switch_notification_shown: bool = False


@dataclass
class ModelUsageInfo:
    """Container for per-model usage details."""

    used: int
    total: int
    input_tokens: int | None = None
    output_tokens: int | None = None


def resolve_timezone(tz_name: str | None) -> ZoneInfo:
    """Return a ``ZoneInfo`` object for the given timezone name.

    If ``tz_name`` is ``None`` or invalid, attempt to use the system timezone
    (``"localtime"``). If that lookup fails, fall back to Pacific Standard Time
    (``"America/Los_Angeles"``).
    """
    if tz_name:
        try:
            return ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            print(f"Warning: Unknown timezone '{tz_name}', using system timezone")
    try:
        return ZoneInfo("localtime")
    except ZoneInfoNotFoundError:
        print("Warning: Unable to detect system timezone, using PST")
        return ZoneInfo("America/Los_Angeles")


def run_ccusage():
    """Execute ccusage blocks --json command and return parsed JSON data."""
    try:
        result = subprocess.run(
            ["ccusage", "blocks", "--json"], capture_output=True, text=True, check=True
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running ccusage: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None


def run_ccusage_session():
    """Execute ccusage session --breakdown --json and return parsed JSON."""
    try:
        result = subprocess.run(
            ["ccusage", "session", "--breakdown", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Error getting session breakdown: {e}")
        return None


def _find_target_session(active_block, sessions):
    """Return the session matching the active block."""
    active_id = active_block.get("sessionId")
    if active_id:
        for sess in sessions:
            if sess.get("sessionId") == active_id:
                return sess
    active_start = active_block.get("startTime")
    active_last = active_block.get("lastActivity")
    for sess in sessions:
        if sess.get("startTime") == active_start or sess.get("lastActivity") == active_last:
            return sess
    if sessions:
        return sorted(
            sessions,
            key=lambda x: x.get("sessionId") or x.get("lastActivity") or "",
            reverse=True,
        )[0]
    return None


def get_session_model_usage(active_block, session_info):
    """Return model token usage mapping for the active session.

    The returned dict maps model name to a dictionary containing:
    ``"total"`` - total tokens, ``"input_tokens"`` and ``"output_tokens"`` when
    available. Older versions of ``ccusage`` may not provide the detailed
    ``inputTokens``/``outputTokens`` fields, in which case those values will be
    ``None``.
    """
    if not session_info or "sessions" not in session_info or not active_block:
        return {}
    sessions = session_info["sessions"]
    target_session = _find_target_session(active_block, sessions)
    model_usage = {}
    if target_session and "modelBreakdowns" in target_session:
        for br in target_session["modelBreakdowns"]:
            model = br.get("model")
            total = br.get("totalTokens", br.get("total", 0))
            input_tokens = br.get("inputTokens")
            output_tokens = br.get("outputTokens")
            if model:
                model_usage[model] = {
                    "total": total,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
    return model_usage


def format_time(minutes):
    """Format minutes into human-readable time (e.g., '3h 45m')."""
    if minutes < 60:
        return f"{int(minutes)}m"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    if mins == 0:
        return f"{hours}h"
    return f"{hours}h {mins}m"


def create_token_progress_bar(percentage, width=50, plain=False):
    """Return a token usage progress bar."""
    if plain or not RICH_AVAILABLE:
        filled = int(width * percentage / 100)
        green_bar = "█" * filled
        red_bar = "░" * (width - filled)
        return (
            f"🟢 [{COLOR_GREEN}{green_bar}{COLOR_RED}{red_bar}{COLOR_RESET}]" f" {percentage:.1f}%"
        )
    progress = Progress(
        BarColumn(bar_width=width, complete_style="bright_green"),
        TextColumn("{task.percentage:>5.1f}%"),
        expand=False,
    )
    progress.add_task("", total=100, completed=int(percentage))
    return progress


def get_model_color(model):
    """Return rich and plain color for a model name."""
    if "opus" in model:
        return ("cyan", COLOR_CYAN)
    if "sonnet" in model:
        return ("green", COLOR_GREEN)
    return ("magenta", "\033[95m")


def create_model_ratio_bar(model_usage, width=40, plain=False):
    """Return a single bar visualizing token ratio across models."""
    total_tokens = sum(v["total"] for v in model_usage.values())
    if total_tokens <= 0:
        return Text("") if not plain and RICH_AVAILABLE else ""

    def _add_segment(model, data, is_last, allocated):
        percentage = (data["total"] / total_tokens) * 100
        bar_len = width - allocated if is_last else int(width * percentage / 100)
        color_name, color_code = get_model_color(model)
        token_segment = "█" * bar_len
        if plain or not RICH_AVAILABLE:
            segments.append(f"{color_code}{token_segment}")
        else:
            segments.append(Text(token_segment, style=color_name))
        info_parts.append(f"{format_model_name(model)} {percentage:.0f}%")
        return bar_len

    segments = []
    info_parts = []
    allocated = 0
    items = list(model_usage.items())
    for idx, (model, data) in enumerate(items):
        allocated += _add_segment(
            model,
            data,
            idx == len(items) - 1,
            allocated,
        )

    if plain or not RICH_AVAILABLE:
        ratio_bar = "".join(segments) + COLOR_RESET
        return f"{ratio_bar} {' '.join(info_parts)}"

    text = Text.assemble(*segments)
    text.append(" " + " ".join(info_parts))
    return text


def create_time_progress_bar(elapsed_minutes, total_minutes, width=50, plain=False):
    """Create a time progress bar showing time until reset."""
    percentage = 0 if total_minutes <= 0 else min(100, (elapsed_minutes / total_minutes) * 100)
    if plain or not RICH_AVAILABLE:
        filled = int(width * percentage / 100)
        blue_bar = "█" * filled
        red_bar = "░" * (width - filled)
        remaining_time = format_time(max(0, total_minutes - elapsed_minutes))
        return f"⏰ [{COLOR_BLUE}{blue_bar}{COLOR_RED}{red_bar}{COLOR_RESET}]" f" {remaining_time}"

    remaining_time = format_time(max(0, total_minutes - elapsed_minutes))
    progress = Progress(
        BarColumn(bar_width=width, complete_style="bright_blue"),
        TextColumn(remaining_time),
        expand=False,
    )
    progress.add_task("", total=total_minutes, completed=elapsed_minutes)
    return progress


def create_model_progress_bar(
    model: str,
    usage: ModelUsageInfo,
    width: int = 40,
    plain: bool = False,
):
    """Return a per-model progress bar including token and cost summary."""
    percentage = (usage.used / usage.total * 100) if usage.total > 0 else 0.0
    summary = format_model_usage(
        model,
        usage.used,
        usage.total,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )
    if plain or not RICH_AVAILABLE:
        filled = int(width * percentage / 100)
        green_bar = "█" * filled
        red_bar = "░" * (width - filled)
        return (
            f"{model:<15} " f"[{COLOR_GREEN}{green_bar}{COLOR_RED}{red_bar}{COLOR_RESET}] {summary}"
        )

    progress = Progress(
        TextColumn(f"{model:<15}"),
        BarColumn(bar_width=width, complete_style="bright_green"),
        TextColumn(summary),
        expand=False,
    )
    progress.add_task("", total=100, completed=int(percentage))
    return progress


# Pricing data source used by the upstream `ccusage` project
LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)

# Fallback pricing in case fetching from LiteLLM fails
DEFAULT_MODEL_PRICING = {
    "claude-opus-4": {
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
    },
    "claude-sonnet-4": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
    "claude-opus-4-20250514": {
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
    },
    "claude-4-opus-20250514": {
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
    },
    "claude-sonnet-4-20250514": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
    "claude-4-sonnet-20250514": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
}

_PRICING_CACHE: dict | None = None


def _fetch_pricing_cache() -> dict:
    global _PRICING_CACHE  # pylint: disable=global-statement
    if _PRICING_CACHE is None:
        try:
            with urllib.request.urlopen(LITELLM_PRICING_URL, timeout=5) as resp:
                _PRICING_CACHE = json.load(resp)
        except (  # pylint: disable=broad-exception-caught
            URLError,
            json.JSONDecodeError,
            Exception,
        ):
            _PRICING_CACHE = DEFAULT_MODEL_PRICING.copy()
    return _PRICING_CACHE


def get_model_pricing(model: str) -> dict | None:
    """Fetch pricing information for a model from LiteLLM."""
    pricing_cache = _fetch_pricing_cache()
    pricing = pricing_cache.get(model)
    if pricing:
        return pricing
    for key, value in pricing_cache.items():
        if model in key:
            return value
    return DEFAULT_MODEL_PRICING.get(model)


def format_model_usage(
    model,
    tokens,
    total_tokens,
    *,
    input_tokens=None,
    output_tokens=None,
):
    """Return formatted percentage, tokens and cost for a model."""

    percentage = (tokens / total_tokens * 100) if total_tokens > 0 else 0.0
    pricing = get_model_pricing(model)
    input_cost = output_cost = 0
    if pricing:
        input_cost = pricing.get("input_cost_per_token", 0)
        output_cost = pricing.get("output_cost_per_token", 0)

    if input_tokens is not None and output_tokens is not None:
        cost = input_tokens * input_cost + output_tokens * output_cost
    else:
        if pricing:
            if input_cost and output_cost:
                cost_per_token = (input_cost + output_cost) / 2
            else:
                cost_per_token = input_cost or output_cost
        else:
            cost_per_token = 0
        cost = tokens * cost_per_token

    return f"{percentage:5.1f}% {tokens:,} tokens (${cost:.2f})"


def format_model_name(model: str | None) -> str | None:
    """Return a short human readable model name."""
    if not model:
        return None
    if "opus" in model:
        return "Opus"
    if "sonnet" in model:
        return "Sonnet"
    return model


def print_header(active_model: str | None = None):
    """Print the stylized header with sparkles."""
    # Sparkle pattern
    sparkles = f"{COLOR_CYAN}✦ ✧ ✦ ✧ {COLOR_RESET}"

    print(f"{sparkles}{COLOR_CYAN}CLAUDE TOKEN MONITOR{COLOR_RESET} {sparkles}")
    print(f"{COLOR_BLUE}{'=' * 60}{COLOR_RESET}")
    if active_model:
        formatted = format_model_name(active_model)
        print(f"Active Model: {formatted}")
    print()


def get_velocity_indicator(burn_rate):
    """Get velocity emoji based on burn rate."""
    if burn_rate < 50:
        return "🐌"  # Slow
    if burn_rate < 150:
        return "➡️"  # Normal
    if burn_rate < 300:
        return "🚀"  # Fast
    return "⚡"  # Very fast


def calculate_hourly_burn_rate(blocks, current_time):
    """Calculate burn rate based on all sessions in the last hour."""
    if not blocks:
        return 0

    one_hour_ago = current_time - timedelta(hours=1)
    total_tokens = 0

    def _get_session_end(b, now):
        actual_end_str = b.get("actualEndTime")
        if b.get("isActive", False):
            return now
        if actual_end_str:
            return datetime.fromisoformat(actual_end_str.replace("Z", "+00:00"))
        return now

    for block in blocks:
        start_time_str = block.get("startTime")
        if not start_time_str or block.get("isGap", False):
            continue

        start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        session_actual_end = _get_session_end(block, current_time)

        # Check if session overlaps with the last hour
        if session_actual_end < one_hour_ago:
            # Session ended before the last hour
            continue

        # Calculate how much of this session falls within the last hour
        session_start_in_hour = max(start_time, one_hour_ago)
        session_end_in_hour = min(session_actual_end, current_time)

        if session_end_in_hour <= session_start_in_hour:
            continue

        # Calculate portion of tokens used in the last hour
        total_session_duration = (session_actual_end - start_time).total_seconds() / 60  # minutes
        hour_duration = (
            session_end_in_hour - session_start_in_hour
        ).total_seconds() / 60  # minutes

        if total_session_duration > 0:
            session_tokens = block.get("totalTokens", 0)
            tokens_in_hour = session_tokens * (hour_duration / total_session_duration)
            total_tokens += tokens_in_hour

    # Return tokens per minute
    return total_tokens / 60 if total_tokens > 0 else 0


def get_next_reset_time(current_time, custom_reset_hour=None, timezone_str=None):
    """Calculate next token reset time based on fixed 5-hour intervals.
    Default reset times in specified timezone: 04:00, 09:00, 14:00, 18:00, 23:00
    Or use custom reset hour if provided.
    """
    # Convert to specified timezone
    target_tz = resolve_timezone(timezone_str)

    # If current_time is timezone-aware, convert to target timezone
    if current_time.tzinfo is not None:
        target_time = current_time.astimezone(target_tz)
    else:
        # Assume current_time is in target timezone if not specified
        target_time = current_time.replace(tzinfo=target_tz)

    if custom_reset_hour is not None:
        # Use single daily reset at custom hour
        reset_hours = [custom_reset_hour]
    else:
        # Default 5-hour intervals
        reset_hours = [4, 9, 14, 18, 23]

    # Get current hour and minute
    current_hour = target_time.hour
    current_minute = target_time.minute

    # Find next reset hour
    next_reset_hour = None
    for hour in reset_hours:
        if current_hour < hour or (current_hour == hour and current_minute == 0):
            next_reset_hour = hour
            break

    # If no reset hour found today, use first one tomorrow
    if next_reset_hour is None:
        next_reset_hour = reset_hours[0]
        next_reset_date = target_time.date() + timedelta(days=1)
    else:
        next_reset_date = target_time.date()

    # Create next reset datetime in target timezone
    next_reset = datetime.combine(
        next_reset_date,
        datetime.min.time().replace(hour=next_reset_hour, tzinfo=target_tz),
    )

    # Convert back to the original timezone if needed
    if current_time.tzinfo is not None and current_time.tzinfo != target_tz:
        next_reset = next_reset.astimezone(current_time.tzinfo)

    return next_reset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Claude Token Monitor - Real-time token usage monitoring"
    )
    parser.add_argument(
        "--plan",
        type=str,
        default="pro",
        choices=["pro", "max5", "max20", "custom_max"],
        help=(
            "Claude plan type (default: pro). "
            'Use "custom_max" to auto-detect from highest previous block'
        ),
    )
    parser.add_argument(
        "--reset-hour", type=int, help="Change the reset hour (0-23) for daily limits"
    )
    parser.add_argument(
        "--timezone",
        type=str,
        help=(
            "Timezone for reset times. Defaults to the system timezone, "
            "falling back to Pacific Standard Time if detection fails."
        ),
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Disable rich output and use simple text",
    )
    return parser.parse_args()


def get_token_limit(plan, blocks=None):
    """Get token limit based on plan type."""
    if plan == "custom_max" and blocks:
        # Find the highest token count from all previous blocks
        max_tokens = 0
        for block in blocks:
            if not block.get("isGap", False) and not block.get("isActive", False):
                tokens = block.get("totalTokens", 0)
                max_tokens = max(max_tokens, tokens)
        # Return the highest found, or default to pro if none found
        return max_tokens if max_tokens > 0 else 7000

    limits = {"pro": 7000, "max5": 35000, "max20": 140000}
    return limits.get(plan, 7000)


def update_switch_state(
    tokens_used: int, plan: str, blocks, state: MonitorState
) -> tuple[str, bool]:
    """Update plan based on usage and determine notification state."""
    if tokens_used > state.token_limit and plan == "pro":
        new_limit = get_token_limit("custom_max", blocks)
        if new_limit > state.token_limit:
            state.token_limit = new_limit
            plan = "custom_max"
            if not state.switched_to_custom_max:
                state.switched_to_custom_max = True

    show = state.switched_to_custom_max and not state.switch_notification_shown
    if show:
        state.switch_notification_shown = True

    return plan, show


def find_active_block(blocks):
    """Return the active block from a list of blocks."""
    for block in blocks:
        if block.get("isActive", False):
            return block
    return None


def collect_session_stats(  # pylint: disable=too-many-locals
    args,
    data_blocks,
    session_info,
    state: MonitorState,
) -> tuple[dict | None, bool]:
    """Return metrics and updated state for the active session."""
    active_block = find_active_block(data_blocks)
    if not active_block:
        return None, False

    tokens_used = active_block.get("totalTokens", 0)

    args.plan, show_switch_notification = update_switch_state(
        tokens_used, args.plan, data_blocks, state
    )

    active_model = active_block.get("model")
    model_usage = get_session_model_usage(active_block, session_info)

    start_time_str = active_block.get("startTime")
    if start_time_str:
        start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        current_time = datetime.now(start_time.tzinfo)
    else:
        current_time = datetime.now()

    burn_rate = calculate_hourly_burn_rate(data_blocks, current_time)
    reset_time = get_next_reset_time(current_time, args.reset_hour, args.timezone)
    minutes_to_reset = (reset_time - current_time).total_seconds() / 60

    tokens_left = state.token_limit - tokens_used
    if burn_rate > 0 and tokens_left > 0:
        predicted_end_time = current_time + timedelta(minutes=tokens_left / burn_rate)
    else:
        predicted_end_time = reset_time

    time_since_reset = max(0, 300 - minutes_to_reset)

    metrics = {
        "tokens_used": tokens_used,
        "active_model": active_model,
        "model_usage": model_usage,
        "burn_rate": burn_rate,
        "reset_time": reset_time,
        "predicted_end_time": predicted_end_time,
        "time_since_reset": time_since_reset,
        "tokens_left": tokens_left,
        "current_time": current_time,
        "show_switch_notification": show_switch_notification,
    }
    return metrics, True


def display_plain_report(metrics, token_limit, args):
    """Print a plain text status update using collected metrics."""
    usage_percentage = metrics["tokens_used"] / token_limit * 100 if token_limit > 0 else 0

    print_header(metrics["active_model"])

    print(
        f"📊 {COLOR_WHITE}Token Usage:{COLOR_RESET}    "
        f"{create_token_progress_bar(usage_percentage, plain=True)}"
    )
    print()

    print(
        f"⏳ {COLOR_WHITE}Time to Reset:{COLOR_RESET}  "
        f"{create_time_progress_bar(metrics['time_since_reset'], 300, plain=True)}"
    )
    print()

    print(
        f"🎯 {COLOR_WHITE}Tokens:{COLOR_RESET}         "
        f"{COLOR_WHITE}{metrics['tokens_used']:,}{COLOR_RESET} / "
        f"{COLOR_GRAY}~{token_limit:,}{COLOR_RESET}"
        f" ({COLOR_CYAN}{metrics['tokens_left']:,} left{COLOR_RESET})"
    )
    print(
        f"🔥 {COLOR_WHITE}Burn Rate:{COLOR_RESET}      "
        f"{COLOR_YELLOW}{metrics['burn_rate']:.1f}{COLOR_RESET} {COLOR_GRAY}tokens/min{COLOR_RESET}"
    )
    if metrics["model_usage"] and len(metrics["model_usage"]) > 1:
        ratio_bar = create_model_ratio_bar(metrics["model_usage"], plain=True)
        print(f"💠 {ratio_bar}")
        print()
    elif metrics["model_usage"]:
        print("\n💠 Model Usage:")
        total_models_tokens = sum(v["total"] for v in metrics["model_usage"].values())
        for m, md in metrics["model_usage"].items():
            usage_info = ModelUsageInfo(
                used=md["total"],
                total=total_models_tokens,
                input_tokens=md.get("input_tokens"),
                output_tokens=md.get("output_tokens"),
            )
            progress_line = create_model_progress_bar(m, usage_info, plain=True)
            print(f"    {progress_line}")
        print()
    else:
        print()

    local_tz = resolve_timezone(args.timezone)
    tz_name = getattr(local_tz, "key", str(local_tz))
    predicted_end_str = metrics["predicted_end_time"].astimezone(local_tz).strftime("%H:%M")
    reset_time_str = metrics["reset_time"].astimezone(local_tz).strftime("%H:%M")
    print(f"🏁 {COLOR_WHITE}Predicted End:{COLOR_RESET} {predicted_end_str}")
    print(f"🔄 {COLOR_WHITE}Token Reset:{COLOR_RESET}   {reset_time_str}")
    print(f"🕒 {COLOR_WHITE}Time Zone:{COLOR_RESET}  {tz_name}")
    print()

    if metrics["show_switch_notification"]:
        print(
            f"🔄 {COLOR_YELLOW}Tokens exceeded Pro limit - switched to "
            f"custom_max ({token_limit:,}){COLOR_RESET}"
        )
        print()
    if metrics["tokens_used"] > token_limit:
        print(
            f"🚨 {COLOR_RED}TOKENS EXCEEDED MAX LIMIT! "
            f"({metrics['tokens_used']:,} > {token_limit:,}){COLOR_RESET}"
        )
        print()
    if metrics["predicted_end_time"] < metrics["reset_time"]:
        print(f"⚠️  {COLOR_RED}Tokens will run out BEFORE reset!{COLOR_RESET}")
        print()
    current_time_str = datetime.now().strftime("%H:%M:%S")
    print(
        (
            f"⏰ {COLOR_GRAY}{current_time_str}{COLOR_RESET} 📝 "
            f"{COLOR_CYAN}Smooth sailing...{COLOR_RESET} | "
            f"{COLOR_GRAY}Ctrl+C to exit{COLOR_RESET} 🟨"
        )
    )

    print("\033[J", end="", flush=True)


def build_rich_panel(metrics, token_limit, args):  # pylint: disable=too-many-locals
    """Return a rich Panel object for the current metrics."""
    tokens_used = metrics["tokens_used"]
    active_model = metrics["active_model"]
    model_usage = metrics["model_usage"]
    burn_rate = metrics["burn_rate"]
    reset_time = metrics["reset_time"]
    predicted_end_time = metrics["predicted_end_time"]
    time_since_reset = metrics["time_since_reset"]
    show_switch_notification = metrics["show_switch_notification"]

    usage_percentage = (tokens_used / token_limit) * 100 if token_limit > 0 else 0

    local_tz = resolve_timezone(args.timezone)
    predicted_end_str = predicted_end_time.astimezone(local_tz).strftime("%H:%M")
    reset_time_str = reset_time.astimezone(local_tz).strftime("%H:%M")

    body = [Text("CLAUDE TOKEN MONITOR", style="bold cyan")]
    if active_model:
        body.append(Text(f"Active Model: {format_model_name(active_model)}"))

    body.append(Text("📊 Token Usage:", style="bold"))
    body.append(create_token_progress_bar(usage_percentage))

    body.append(Text("⏳ Time to Reset:", style="bold"))
    body.append(create_time_progress_bar(time_since_reset, 300))

    body.append(
        Text(
            f"🎯 Tokens: {tokens_used:,} / ~{token_limit:,} ({metrics['tokens_left']:,} left)",
            style="white",
        )
    )
    body.append(Text(f"🔥 Burn Rate: {burn_rate:.1f} tokens/min", style="yellow"))

    if model_usage:
        if len(model_usage) > 1:
            body.append(create_model_ratio_bar(model_usage))
            body.append(Text(""))
        else:
            body.append(Text("\n💠 Model Usage:", style="bold"))
            total_tokens = sum(v["total"] for v in model_usage.values())
            for m, md in model_usage.items():
                usage_info = ModelUsageInfo(
                    used=md["total"],
                    total=total_tokens,
                    input_tokens=md.get("input_tokens"),
                    output_tokens=md.get("output_tokens"),
                )
                body.append(create_model_progress_bar(m, usage_info))
            body.append(Text(""))

    tz_name = getattr(local_tz, "key", str(local_tz))
    body.append(Text(f"🏁 Predicted End: {predicted_end_str}"))
    body.append(Text(f"🔄 Token Reset:   {reset_time_str}"))
    body.append(Text(f"🕒 Time Zone:  {tz_name}"))

    if show_switch_notification:
        body.append(
            Text(
                f"🔄 Tokens exceeded Pro limit - switched to custom_max ({token_limit:,})",
                style="yellow",
            )
        )
    if tokens_used > token_limit:
        body.append(
            Text(
                f"🚨 TOKENS EXCEEDED MAX LIMIT! ({tokens_used:,} > {token_limit:,})",
                style="red",
            )
        )
    if predicted_end_time < reset_time:
        body.append(Text("⚠️  Tokens will run out BEFORE reset!", style="red"))

    current_time_str = datetime.now().strftime("%H:%M:%S")
    body.append(
        Text(
            f"⏰ {current_time_str} 📝 Smooth sailing... | Ctrl+C to exit 🟨",
            style="dim",
        )
    )

    return Panel(Group(*body))


def run_plain_once(args, data, session_info, state: MonitorState) -> None:
    """Render a single plain-text update."""
    if not data or "blocks" not in data:
        print("Failed to get usage data")
        return

    metrics, ok = collect_session_stats(
        args,
        data["blocks"],
        session_info,
        state,
    )

    if not ok or metrics is None:
        print("No active session found")
        return

    display_plain_report(metrics, state.token_limit, args)

    return


def run_plain(args, token_limit):
    """Main monitoring loop using plain text output."""
    state = MonitorState(token_limit)

    try:
        os.system("clear" if os.name == "posix" else "cls")
        print("\033[?25l", end="", flush=True)

        while True:
            print("\033[H", end="", flush=True)
            data = run_ccusage()
            session_info = run_ccusage_session()
            run_plain_once(args, data, session_info, state)
            time.sleep(3)

    except KeyboardInterrupt:
        print("\033[?25h", end="", flush=True)
        print(f"\n\n{COLOR_CYAN}Monitoring stopped.{COLOR_RESET}")
        os.system("clear" if os.name == "posix" else "cls")
        sys.exit(0)


def run_rich_once(
    args,
    data,
    session_info,
    console: Console,
    state: MonitorState,
) -> Panel | None:
    """Render a single rich update and return updated state."""
    if not data or "blocks" not in data:
        console.print("Failed to get usage data")
        return None

    metrics, ok = collect_session_stats(
        args,
        data["blocks"],
        session_info,
        state,
    )

    if not ok or metrics is None:
        console.print("No active session found")
        return None

    panel = build_rich_panel(metrics, state.token_limit, args)
    console.print(panel)
    return panel


def run_rich(args, token_limit):
    """Monitoring loop using rich output."""
    state = MonitorState(token_limit)
    console = Console()
    with Live(console=console, refresh_per_second=4, screen=True) as live:
        try:
            while True:
                data = run_ccusage()
                session_info = run_ccusage_session()
                panel = run_rich_once(
                    args,
                    data,
                    session_info,
                    console=console,
                    state=state,
                )
                if panel is not None:
                    live.update(panel)
                time.sleep(3)
        except KeyboardInterrupt:
            console.print("\nMonitoring stopped.", style="cyan")


def main():
    args = parse_args()
    if args.plan == "custom_max":
        initial_data = run_ccusage()
        if initial_data and "blocks" in initial_data:
            token_limit = get_token_limit(args.plan, initial_data["blocks"])
        else:
            token_limit = get_token_limit("pro")
    else:
        token_limit = get_token_limit(args.plan)

    if args.plain or not RICH_AVAILABLE:
        run_plain(args, token_limit)
    else:
        run_rich(args, token_limit)


if __name__ == "__main__":
    main()
