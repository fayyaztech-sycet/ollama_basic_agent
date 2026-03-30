import json
from datetime import datetime

# Token usage tracking with cost calculation
TOKEN_LOG_PATH = os.path.join(HOME_DIR, ".ollama_agent_token_log.json")
# Set your cost per 1K tokens (USD)
IN_TOKEN_COST_PER_1K = 0.002  # Example: $0.002 per 1K input tokens
OUT_TOKEN_COST_PER_1K = 0.002  # Example: $0.002 per 1K output tokens

def log_token_usage(input_tokens: int, output_tokens: int) -> None:
    """Log input/output token usage for today."""
    today = datetime.now().strftime("%Y-%m-%d")
    data = {}
    if os.path.exists(TOKEN_LOG_PATH):
        try:
            with open(TOKEN_LOG_PATH, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
    if today not in data:
        data[today] = {"input": 0, "output": 0}
    data[today]["input"] += input_tokens
    data[today]["output"] += output_tokens
    with open(TOKEN_LOG_PATH, "w") as f:
        json.dump(data, f)

def get_token_dashboard(days: int = 7) -> str:
    """Return a dashboard of daily token usage and cost for the last N days."""
    if not os.path.exists(TOKEN_LOG_PATH):
        return "No token usage data found."
    try:
        with open(TOKEN_LOG_PATH, "r") as f:
            data = json.load(f)
        # Sort by date descending
        items = sorted(data.items(), reverse=True)[:days]
        lines = ["| Date       | Input | Output | Total | Cost (USD) |", "|------------|-------|--------|-------|------------|"]
        for date, usage in items:
            input_tokens = usage.get("input", 0)
            output_tokens = usage.get("output", 0)
            total = input_tokens + output_tokens
            cost = (input_tokens / 1000) * IN_TOKEN_COST_PER_1K + (output_tokens / 1000) * OUT_TOKEN_COST_PER_1K
            lines.append(f"| {date} | {input_tokens:,} | {output_tokens:,} | {total:,} | ${cost:.4f}     |")
        return "\n".join(lines)
    except Exception as e:
        return f"Error reading token dashboard: {e}"
import os
import logging
import subprocess
import psutil

logger = logging.getLogger("agent")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

HOME_DIR = os.path.expanduser("~")

# Commands that are allowed to run
WHITELISTED_CMDS = {
    "nvidia-smi", "whoami", "uptime", "df",
    "ls", "cat", "yt-dlp", "ffmpeg", "ffprobe",
    "date", "hostname", "uname",   # safe read-only system info
}

# Commands restricted to home directory paths only
HOME_RESTRICTED_CMDS = {"cat", "ls"}

# Subprocess timeout in seconds
CMD_TIMEOUT = 300   # 5 min — covers long yt-dlp / ffmpeg jobs


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _safe_path(raw: str) -> str:
    """Expand ~ and resolve to an absolute path."""
    return os.path.abspath(os.path.expanduser(raw))


def _assert_home(path: str, cmd: str) -> str | None:
    """
    Return an error string if `path` escapes the home directory.
    Returns None if the path is safe.
    """
    if not path.startswith(HOME_DIR):
        return (
            f"Error: '{cmd}' is restricted to your home directory. "
            f"Requested path '{path}' is outside '{HOME_DIR}'."
        )
    return None


def _run(cmd: list, timeout: int = CMD_TIMEOUT) -> tuple[int, str, str]:
    """
    Run a subprocess safely.
    Returns (returncode, stdout, stderr).
    """
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

def open_file(path: str) -> str:
    """Open a file or directory using the system default handler (xdg-open)."""
    try:
        abs_path = _safe_path(path)

        if not os.path.exists(abs_path):
            return f"Error: Path '{abs_path}' does not exist."

        subprocess.Popen(
            ["xdg-open", abs_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info(f"Opened: '{abs_path}'")
        return f"Successfully opened '{abs_path}'."

    except FileNotFoundError:
        return "Error: 'xdg-open' is not available on this system."
    except Exception as e:
        logger.exception(f"open_file failed for '{path}': {e}")
        return f"Error opening '{path}': {e}"


def list_directory(
    path: str = ".",
    show_sizes: bool = False,
    include_dir_size: bool = False
) -> str:
    """List files and folders with optional size info (read-only, safe)."""
    try:
        abs_path = _safe_path(path)

        if not os.path.exists(abs_path):
            return f"Error: Path '{abs_path}' does not exist."
        if not os.path.isdir(abs_path):
            return f"Error: '{abs_path}' is not a directory."

        items = sorted(os.listdir(abs_path))   # sorted for consistent output
        if not items:
            return "Directory is empty."

        result = []
        for item in items:
            full_path = os.path.join(abs_path, item)
            try:
                if os.path.isdir(full_path):
                    if show_sizes and include_dir_size:
                        size = _get_dir_size(full_path)
                        result.append(f"[DIR]  {item}  ({size:,} bytes)")
                    else:
                        result.append(f"[DIR]  {item}")
                else:
                    if show_sizes:
                        size = os.path.getsize(full_path)
                        result.append(f"[FILE] {item}  ({size:,} bytes)")
                    else:
                        result.append(f"[FILE] {item}")
            except OSError as e:
                result.append(f"[????] {item}  (unreadable: {e})")

        logger.info(f"Listed directory: '{abs_path}' ({len(result)} items)")
        return "\n".join(result)

    except PermissionError:
        return f"Error: Permission denied reading '{path}'."
    except Exception as e:
        logger.exception(f"list_directory failed for '{path}': {e}")
        return f"Error listing directory: {e}"


def _get_dir_size(path: str) -> int:
    """Recursively calculate total size of a directory in bytes."""
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except (OSError, PermissionError):
                pass   # skip unreadable files silently
    return total


def run_safe_command(base_cmd: str, *args) -> str:
    """
    Run a whitelisted shell command with optional arguments.
    'cat' and 'ls' are restricted to the user's home directory.
    """
    if base_cmd not in WHITELISTED_CMDS:
        available = ", ".join(sorted(WHITELISTED_CMDS))
        return (
            f"Error: '{base_cmd}' is not in the safe whitelist. "
            f"Available commands: {available}"
        )

    # Expand ~ in all string arguments
    expanded_args = [
        _safe_path(a) if isinstance(a, str) else a
        for a in args
    ]

    # Home-directory restriction for sensitive read commands
    if base_cmd in HOME_RESTRICTED_CMDS and expanded_args:
        target = expanded_args[0]
        err = _assert_home(target, base_cmd)
        if err:
            return err

    try:
        cmd = [base_cmd] + [str(a) for a in expanded_args]
        logger.info(f"run_safe_command: {cmd}")
        rc, stdout, stderr = _run(cmd)

        if rc == 0:
            return stdout if stdout else "Command executed successfully (no output)."
        else:
            logger.warning(f"Command '{base_cmd}' exited {rc}: {stderr}")
            return f"Error running '{base_cmd}' (exit {rc}): {stderr}"

    except subprocess.TimeoutExpired:
        logger.error(f"Command '{base_cmd}' timed out after {CMD_TIMEOUT}s.")
        return f"Error: '{base_cmd}' timed out after {CMD_TIMEOUT} seconds."
    except FileNotFoundError:
        return (
            f"Error: '{base_cmd}' is not installed or not found in PATH. "
            "Please install it and try again."
        )
    except Exception as e:
        logger.exception(f"run_safe_command failed [{base_cmd}]: {e}")
        return f"Unexpected error running '{base_cmd}': {e}"


def gpu_status() -> str:
    """Return GPU usage via nvidia-smi."""
    return run_safe_command("nvidia-smi")


def get_system_status() -> str:
    """Return CPU %, RAM usage, and top processes by CPU and memory."""
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        ram       = psutil.virtual_memory()

        processes = []
        for proc in psutil.process_iter(["name", "cpu_percent", "memory_percent"]):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        top_cpu = sorted(processes, key=lambda p: p["cpu_percent"],    reverse=True)[:3]
        top_mem = sorted(processes, key=lambda p: p["memory_percent"], reverse=True)[:3]

        status = {
            "cpu_percent":       cpu_usage,
            "ram_total_gb":      round(ram.total / (1024 ** 3), 2),
            "ram_used_gb":       round(ram.used  / (1024 ** 3), 2),
            "ram_percent":       ram.percent,
            "top_processes_cpu": top_cpu,
            "top_processes_mem": top_mem,
        }
        logger.info(f"System status: cpu={cpu_usage}% ram={ram.percent}%")
        return str(status)

    except Exception as e:
        logger.exception(f"get_system_status failed: {e}")
        return f"Error getting system status: {e}"


def check_updates() -> str:
    """Check for upgradable packages via apt (Debian/Ubuntu)."""
    try:
        rc, stdout, stderr = _run(["apt", "list", "--upgradable"], timeout=30)

        if rc != 0:
            return (
                f"Could not check for updates (apt exited {rc}). "
                f"You may need sudo privileges. Details: {stderr}"
            )

        upgradable = [l for l in stdout.splitlines() if "/" in l]
        count = len(upgradable)

        if count == 0:
            return "Your system is up to date."

        examples  = ", ".join(pkg.split("/")[0] for pkg in upgradable[:5])
        summary   = f"Found {count} upgradable package(s). Examples: {examples}"
        if count > 5:
            summary += f" ... and {count - 5} more."
        logger.info(f"check_updates: {count} upgradable packages.")
        return summary

    except subprocess.TimeoutExpired:
        return "Error: apt timed out while checking for updates."
    except FileNotFoundError:
        return "Error: 'apt' is not available. This tool requires a Debian/Ubuntu system."
    except Exception as e:
        logger.exception(f"check_updates failed: {e}")
        return f"Error checking updates: {e}"


def download_youtube(url: str) -> str:
    """
    Download a YouTube video using yt-dlp.
    Returns the saved filepath so memory tracking works correctly.
    """
    try:
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
            "--no-playlist",
            "--print", "after_move:%(filepath)s",   # prints final path after download
            "-o", os.path.join(HOME_DIR, "%(title)s.%(ext)s"),
            url,
        ]
        logger.info(f"download_youtube: {url}")
        rc, stdout, stderr = _run(cmd)

        if rc == 0:
            # Last non-empty line is the saved filepath
            filepath = stdout.strip().splitlines()[-1] if stdout.strip() else "unknown"
            logger.info(f"Downloaded: '{filepath}'")
            return f"Successfully downloaded as {filepath}"
        else:
            logger.warning(f"yt-dlp failed (exit {rc}): {stderr[:200]}")
            return f"Error downloading video: {stderr}"

    except subprocess.TimeoutExpired:
        return f"Error: yt-dlp timed out after {CMD_TIMEOUT} seconds."
    except FileNotFoundError:
        return "Error: 'yt-dlp' is not installed. Install it with: pip install yt-dlp"
    except Exception as e:
        logger.exception(f"download_youtube failed: {e}")
        return f"Error running yt-dlp: {e}"


def convert_video(input_file: str, output_format: str) -> str:
    """
    Convert a video file to another format using ffmpeg.
    Will NOT overwrite an existing output file.
    """
    try:
        input_file = _safe_path(input_file)

        if not os.path.exists(input_file):
            return f"Error: Input file '{input_file}' does not exist."

        # Sanitise format string — letters and digits only
        output_format = output_format.strip().lstrip(".").lower()
        if not output_format.isalnum():
            return f"Error: Invalid output format '{output_format}'."

        base        = os.path.splitext(input_file)[0]
        output_file = f"{base}.{output_format}"

        if os.path.exists(output_file):
            return (
                f"Error: Output file '{output_file}' already exists. "
                "Rename or remove it first."
            )

        cmd = ["ffmpeg", "-i", input_file, "-n", output_file]
        logger.info(f"convert_video: {input_file} → {output_file}")
        rc, stdout, stderr = _run(cmd)

        if rc == 0:
            logger.info(f"Conversion complete: '{output_file}'")
            return f"Successfully converted to {output_file}"
        else:
            logger.warning(f"ffmpeg failed (exit {rc}): {stderr[:200]}")
            return f"Error converting video: {stderr}"

    except subprocess.TimeoutExpired:
        return f"Error: ffmpeg timed out after {CMD_TIMEOUT} seconds."
    except FileNotFoundError:
        return "Error: 'ffmpeg' is not installed. Install it with: sudo apt install ffmpeg"
    except Exception as e:
        logger.exception(f"convert_video failed: {e}")
        return f"Error running ffmpeg: {e}"


# ─────────────────────────────────────────────
# TOOL REGISTRY
# ─────────────────────────────────────────────

AVAILABLE_TOOLS = {
    "check_updates":    check_updates,
    "download_youtube": download_youtube,
    "convert_video":    convert_video,
    "get_system_status": get_system_status,
    "run_safe_command": run_safe_command,
    "gpu_status":       gpu_status,
    "list_directory":   list_directory,
    "open_file":        open_file,
}