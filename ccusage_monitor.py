#!/usr/bin/env python3

import subprocess
import json
import sys
import time
from datetime import datetime, timedelta
import os
import argparse
import urllib.request
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from rich.console import Console, Group
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text

    RICH_AVAILABLE = True
except Exception:  # pragma: no cover
    RICH_AVAILABLE = False


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


def get_session_model_usage(active_block, session_info):
    """Return {model: total_tokens} mapping for the active session."""
    if not session_info or "sessions" not in session_info or not active_block:
        return {}
    sessions = session_info["sessions"]
    target_session = None
    active_id = active_block.get("sessionId")
    if active_id:
        for s in sessions:
            if s.get("sessionId") == active_id:
                target_session = s
                break
    if not target_session:
        active_start = active_block.get("startTime")
        active_last = active_block.get("lastActivity")
        for s in sessions:
            if (
                s.get("startTime") == active_start
                or s.get("lastActivity") == active_last
            ):
                target_session = s
                break
    if not target_session and sessions:
        target_session = sorted(
            sessions,
            key=lambda x: x.get("sessionId") or x.get("lastActivity") or "",
            reverse=True,
        )[0]
    model_usage = {}
    if target_session and "modelBreakdowns" in target_session:
        for br in target_session["modelBreakdowns"]:
            model = br.get("model")
            total = br.get("totalTokens", br.get("total", 0))
            input_tokens = br.get("inputTokens")
            output_tokens = br.get("outputTokens")
            if model:
                model_usage[model] = {
                    "tokens": total,
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
        green = "\033[92m"
        red = "\033[91m"
        reset = "\033[0m"
        return f"🟢 [{green}{green_bar}{red}{red_bar}{reset}] {percentage:.1f}%"
    progress = Progress(
        BarColumn(bar_width=width, complete_style="bright_green"),
        TextColumn("{task.percentage:>5.1f}%"),
        expand=False,
    )
    progress.add_task("", total=100, completed=percentage)
    return progress


def create_time_progress_bar(elapsed_minutes, total_minutes, width=50, plain=False):
    """Create a time progress bar showing time until reset."""
    percentage = (
        0 if total_minutes <= 0 else min(100, (elapsed_minutes / total_minutes) * 100)
    )
    if plain or not RICH_AVAILABLE:
        filled = int(width * percentage / 100)
        blue_bar = "█" * filled
        red_bar = "░" * (width - filled)
        blue = "\033[94m"
        red = "\033[91m"
        reset = "\033[0m"
        remaining_time = format_time(max(0, total_minutes - elapsed_minutes))
        return f"⏰ [{blue}{blue_bar}{red}{red_bar}{reset}] {remaining_time}"

    remaining_time = format_time(max(0, total_minutes - elapsed_minutes))
    progress = Progress(
        BarColumn(bar_width=width, complete_style="bright_blue"),
        TextColumn(remaining_time),
        expand=False,
    )
    progress.add_task("", total=total_minutes, completed=elapsed_minutes)
    return progress


# Pricing data source used by the upstream `ccusage` project
LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
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


def get_model_pricing(model: str) -> dict | None:
    """Fetch pricing information for a model from LiteLLM."""
    global _PRICING_CACHE
    if _PRICING_CACHE is None:
        try:
            with urllib.request.urlopen(LITELLM_PRICING_URL, timeout=5) as resp:
                _PRICING_CACHE = json.load(resp)
        except Exception:
            _PRICING_CACHE = DEFAULT_MODEL_PRICING.copy()
    pricing = _PRICING_CACHE.get(model)
    if pricing:
        return pricing
    for key, value in _PRICING_CACHE.items():
        if model in key:
            return value
    return DEFAULT_MODEL_PRICING.get(model)


def format_model_usage(
    model, tokens, total_tokens, input_tokens=None, output_tokens=None
):
    """Return formatted percentage, tokens and cost for a model."""
    if tokens is None:
        tokens = (input_tokens or 0) + (output_tokens or 0)

    percentage = (tokens / total_tokens * 100) if total_tokens > 0 else 0.0

    pricing = get_model_pricing(model)
    if pricing:
        input_cost = pricing.get("input_cost_per_token", 0)
        output_cost = pricing.get("output_cost_per_token", 0)
        if input_tokens is not None and output_tokens is not None:
            cost = input_tokens * input_cost + output_tokens * output_cost
        else:
            if input_cost and output_cost:
                cost_per_token = (input_cost + output_cost) / 2
            else:
                cost_per_token = input_cost or output_cost
            cost = tokens * cost_per_token
    else:
        cost = 0

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
    cyan = "\033[96m"
    blue = "\033[94m"
    reset = "\033[0m"

    # Sparkle pattern
    sparkles = f"{cyan}✦ ✧ ✦ ✧ {reset}"

    print(f"{sparkles}{cyan}CLAUDE TOKEN MONITOR{reset} {sparkles}")
    print(f"{blue}{'=' * 60}{reset}")
    if active_model:
        formatted = format_model_name(active_model)
        print(f"Active Model: {formatted}")
    print()


def get_velocity_indicator(burn_rate):
    """Get velocity emoji based on burn rate."""
    if burn_rate < 50:
        return "🐌"  # Slow
    elif burn_rate < 150:
        return "➡️"  # Normal
    elif burn_rate < 300:
        return "🚀"  # Fast
    else:
        return "⚡"  # Very fast


def calculate_hourly_burn_rate(blocks, current_time):
    """Calculate burn rate based on all sessions in the last hour."""
    if not blocks:
        return 0

    one_hour_ago = current_time - timedelta(hours=1)
    total_tokens = 0

    for block in blocks:
        start_time_str = block.get("startTime")
        if not start_time_str:
            continue

        # Parse start time
        start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))

        # Skip gaps
        if block.get("isGap", False):
            continue

        # Determine session end time
        if block.get("isActive", False):
            # For active sessions, use current time
            session_actual_end = current_time
        else:
            # For completed sessions, use actualEndTime or current time
            actual_end_str = block.get("actualEndTime")
            if actual_end_str:
                session_actual_end = datetime.fromisoformat(
                    actual_end_str.replace("Z", "+00:00")
                )
            else:
                session_actual_end = current_time

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
        total_session_duration = (
            session_actual_end - start_time
        ).total_seconds() / 60  # minutes
        hour_duration = (
            session_end_in_hour - session_start_in_hour
        ).total_seconds() / 60  # minutes

        if total_session_duration > 0:
            session_tokens = block.get("totalTokens", 0)
            tokens_in_hour = session_tokens * (hour_duration / total_session_duration)
            total_tokens += tokens_in_hour

    # Return tokens per minute
    return total_tokens / 60 if total_tokens > 0 else 0


def get_next_reset_time(
    current_time, custom_reset_hour=None, timezone_str="Europe/Warsaw"
):
    """Calculate next token reset time based on fixed 5-hour intervals.
    Default reset times in specified timezone: 04:00, 09:00, 14:00, 18:00, 23:00
    Or use custom reset hour if provided.
    """
    # Convert to specified timezone
    try:
        target_tz = ZoneInfo(timezone_str)
    except ZoneInfoNotFoundError:
        print(f"Warning: Unknown timezone '{timezone_str}', using Europe/Warsaw")
        target_tz = ZoneInfo("Europe/Warsaw")

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
        default="Europe/Warsaw",
        help=(
            "Timezone for reset times (default: Europe/Warsaw). "
            "Examples: US/Eastern, Asia/Tokyo, UTC"
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
                if tokens > max_tokens:
                    max_tokens = tokens
        # Return the highest found, or default to pro if none found
        return max_tokens if max_tokens > 0 else 7000

    limits = {"pro": 7000, "max5": 35000, "max20": 140000}
    return limits.get(plan, 7000)


def update_switch_state(tokens_used, token_limit, plan, switched, shown, blocks):
    """Update plan based on usage and determine notification state."""
    if tokens_used > token_limit and plan == "pro":
        new_limit = get_token_limit("custom_max", blocks)
        if new_limit > token_limit:
            token_limit = new_limit
            plan = "custom_max"
            if not switched:
                switched = True

    show = switched and not shown
    if show:
        shown = True

    return plan, token_limit, switched, shown, show


def run_plain(args, token_limit):
    """Main monitoring loop using plain text output."""
    # Track if we've switched from pro to a higher custom limit
    switched_to_custom_max = False
    switch_notification_shown = False

    try:
        # Initial screen clear and hide cursor
        os.system("clear" if os.name == "posix" else "cls")
        print("\033[?25l", end="", flush=True)  # Hide cursor

        while True:
            # Move cursor to top without clearing
            print("\033[H", end="", flush=True)

            data = run_ccusage()
            if not data or "blocks" not in data:
                print("Failed to get usage data")
                continue

            # Find the active block
            active_block = None
            for block in data["blocks"]:
                if block.get("isActive", False):
                    active_block = block
                    break

            if not active_block:
                print("No active session found")
                continue

            # Extract data from active block
            tokens_used = active_block.get("totalTokens", 0)
            active_model = active_block.get("model")

            session_info = run_ccusage_session()
            model_usage = get_session_model_usage(active_block, session_info)

            # Update plan and notification state
            (
                args.plan,
                token_limit,
                switched_to_custom_max,
                switch_notification_shown,
                show_switch_notification,
            ) = update_switch_state(
                tokens_used,
                token_limit,
                args.plan,
                switched_to_custom_max,
                switch_notification_shown,
                data["blocks"],
            )

            usage_percentage = (
                (tokens_used / token_limit) * 100 if token_limit > 0 else 0
            )
            tokens_left = token_limit - tokens_used

            # Time calculations
            start_time_str = active_block.get("startTime")
            if start_time_str:
                start_time = datetime.fromisoformat(
                    start_time_str.replace("Z", "+00:00")
                )
                current_time = datetime.now(start_time.tzinfo)
            else:
                pass

            # Calculate burn rate from ALL sessions in the last hour
            burn_rate = calculate_hourly_burn_rate(data["blocks"], current_time)

            # Reset time calculation - use fixed schedule or custom hour with timezone
            reset_time = get_next_reset_time(
                current_time, args.reset_hour, args.timezone
            )

            # Calculate time to reset
            time_to_reset = reset_time - current_time
            minutes_to_reset = time_to_reset.total_seconds() / 60

            # Predicted end calculation - when tokens will run out based on burn rate
            if burn_rate > 0 and tokens_left > 0:
                minutes_to_depletion = tokens_left / burn_rate
                predicted_end_time = current_time + timedelta(
                    minutes=minutes_to_depletion
                )
            else:
                # If no burn rate or tokens already depleted, use reset time
                predicted_end_time = reset_time

            # Color codes
            cyan = "\033[96m"
            red = "\033[91m"
            yellow = "\033[93m"
            white = "\033[97m"
            gray = "\033[90m"
            reset = "\033[0m"

            # Display header
            print_header(active_model)

            # Token Usage section
            print(
                f"📊 {white}Token Usage:{reset}    "
                f"{create_token_progress_bar(usage_percentage, plain=True)}"
            )
            print()

            # Time to Reset section - calculate progress based on time since last reset
            # Estimate time since last reset (max 5 hours = 300 minutes)
            time_since_reset = max(0, 300 - minutes_to_reset)
            print(
                f"⏳ {white}Time to Reset:{reset}  "
                f"{create_time_progress_bar(time_since_reset, 300, plain=True)}"
            )
            print()

            # Detailed stats
            print(
                f"🎯 {white}Tokens:{reset}         "
                f"{white}{tokens_used:,}{reset} / {gray}~{token_limit:,}{reset}"
                f" ({cyan}{tokens_left:,} left{reset})"
            )
            print(
                f"🔥 {white}Burn Rate:{reset}      "
                f"{yellow}{burn_rate:.1f}{reset} {gray}tokens/min{reset}"
            )
            if model_usage:
                print("\n💠 Model Usage:")
                total_models_tokens = sum(v["tokens"] for v in model_usage.values())
                for m, stats in model_usage.items():
                    summary = format_model_usage(
                        m,
                        stats["tokens"],
                        total_models_tokens,
                        stats.get("input_tokens"),
                        stats.get("output_tokens"),
                    )
                    print(f"    {m:<15} {summary}")
                print()
            else:
                print()

            # Predictions - convert to configured timezone for display
            try:
                local_tz = ZoneInfo(args.timezone)
            except ZoneInfoNotFoundError:
                local_tz = ZoneInfo("Europe/Warsaw")
            predicted_end_local = predicted_end_time.astimezone(local_tz)
            reset_time_local = reset_time.astimezone(local_tz)

            predicted_end_str = predicted_end_local.strftime("%H:%M")
            reset_time_str = reset_time_local.strftime("%H:%M")
            print(f"🏁 {white}Predicted End:{reset} {predicted_end_str}")
            print(f"🔄 {white}Token Reset:{reset}   {reset_time_str}")
            print()

            # Notification when exceeding Pro plan (show once)

            # Notification when tokens exceed max limit
            show_exceed_notification = tokens_used > token_limit

            # Show notifications
            if show_switch_notification:
                print(
                    f"🔄 {yellow}Tokens exceeded Pro limit - switched to "
                    f"custom_max ({token_limit:,}){reset}"
                )
                print()

            if show_exceed_notification:
                print(
                    f"🚨 {red}TOKENS EXCEEDED MAX LIMIT! "
                    f"({tokens_used:,} > {token_limit:,}){reset}"
                )
                print()

            # Warning if tokens will run out before reset
            if predicted_end_time < reset_time:
                print(f"⚠️  {red}Tokens will run out BEFORE reset!{reset}")
                print()

            # Status line
            current_time_str = datetime.now().strftime("%H:%M:%S")
            print(
                f"⏰ {gray}{current_time_str}{reset} 📝 {cyan}Smooth sailing...{reset} | "
                f"{gray}Ctrl+C to exit{reset} 🟨"
            )

            # Clear any remaining lines below to prevent artifacts
            print("\033[J", end="", flush=True)

            time.sleep(3)

    except KeyboardInterrupt:
        # Show cursor before exiting
        print("\033[?25h", end="", flush=True)
        print(f"\n\n{cyan}Monitoring stopped.{reset}")
        # Clear the terminal
        os.system("clear" if os.name == "posix" else "cls")
        sys.exit(0)
    except Exception:
        print("\033[?25h", end="", flush=True)
        raise


def run_rich(args, token_limit):
    """Monitoring loop using rich output."""
    switched_to_custom_max = False
    switch_notification_shown = False
    console = Console()
    with Live(console=console, refresh_per_second=4, screen=True) as live:
        try:
            while True:
                data = run_ccusage()
                if not data or "blocks" not in data:
                    console.print("Failed to get usage data")
                    continue

                active_block = None
                for block in data["blocks"]:
                    if block.get("isActive", False):
                        active_block = block
                        break

                if not active_block:
                    console.print("No active session found")
                    continue

            tokens_used = active_block.get("totalTokens", 0)
            active_model = active_block.get("model")
            session_info = run_ccusage_session()
            model_usage = get_session_model_usage(active_block, session_info)

            (
                args.plan,
                token_limit,
                switched_to_custom_max,
                switch_notification_shown,
                show_switch_notification,
            ) = update_switch_state(
                tokens_used,
                token_limit,
                args.plan,
                switched_to_custom_max,
                switch_notification_shown,
                data["blocks"],
            )

            usage_percentage = (
                (tokens_used / token_limit) * 100 if token_limit > 0 else 0
            )
            tokens_left = token_limit - tokens_used

            start_time_str = active_block.get("startTime")
            if start_time_str:
                start_time = datetime.fromisoformat(
                    start_time_str.replace("Z", "+00:00")
                )
                current_time = datetime.now(start_time.tzinfo)
            else:
                current_time = datetime.now()

            burn_rate = calculate_hourly_burn_rate(data["blocks"], current_time)
            reset_time = get_next_reset_time(
                current_time, args.reset_hour, args.timezone
            )
            time_to_reset = reset_time - current_time
            minutes_to_reset = time_to_reset.total_seconds() / 60

            if burn_rate > 0 and tokens_left > 0:
                minutes_to_depletion = tokens_left / burn_rate
                predicted_end_time = current_time + timedelta(
                    minutes=minutes_to_depletion
                )
            else:
                predicted_end_time = reset_time

            time_since_reset = max(0, 300 - minutes_to_reset)

            try:
                local_tz = ZoneInfo(args.timezone)
            except ZoneInfoNotFoundError:
                local_tz = ZoneInfo("Europe/Warsaw")
            predicted_end_local = predicted_end_time.astimezone(local_tz)
            reset_time_local = reset_time.astimezone(local_tz)
            predicted_end_str = predicted_end_local.strftime("%H:%M")
            reset_time_str = reset_time_local.strftime("%H:%M")

            body = [
                Text("CLAUDE TOKEN MONITOR", style="bold cyan"),
                create_token_progress_bar(usage_percentage),
                create_time_progress_bar(time_since_reset, 300),
                Text(
                    f"🎯 Tokens: {tokens_used:,} / ~{token_limit:,} ({tokens_left:,} left)",
                    style="white",
                ),
                Text(f"🔥 Burn Rate: {burn_rate:.1f} tokens/min", style="yellow"),
            ]

            if active_model:
                body.insert(1, Text(f"Active Model: {format_model_name(active_model)}"))

            if model_usage:
                body.append(Text("\n💠 Model Usage:", style="bold"))
                total_models_tokens = sum(v["tokens"] for v in model_usage.values())
                for m, stats in model_usage.items():
                    summary = format_model_usage(
                        m,
                        stats["tokens"],
                        total_models_tokens,
                        stats.get("input_tokens"),
                        stats.get("output_tokens"),
                    )
                    body.append(Text(f"    {m:<15} {summary}"))
                body.append(Text(""))
            else:
                body.append(Text(""))

            body.append(Text(f"🏁 Predicted End: {predicted_end_str}"))
            body.append(Text(f"🔄 Token Reset:   {reset_time_str}"))

            show_exceed_notification = tokens_used > token_limit
            if show_switch_notification:
                body.append(
                    Text(
                        f"🔄 Tokens exceeded Pro limit - switched to custom_max ({token_limit:,})",
                        style="yellow",
                    )
                )
            if show_exceed_notification:
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

            renderable = Panel(Group(*body))
            live.update(renderable)
            time.sleep(3)
        except KeyboardInterrupt:
            console.print("\nMonitoring stopped.", style="cyan")
        except Exception:
            raise


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
