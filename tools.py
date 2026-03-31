
import os
import re
import glob
import json
import logging
import subprocess
import psutil
import urllib.parse
from datetime import datetime

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

HOME_DIR = os.path.expanduser("~")

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


def sanitize_path(raw: str, allowed_root: str = HOME_DIR) -> tuple[str, str | None]:
    """
    Resolve path and verify it stays within allowed_root (blocks path traversal).
    Returns (resolved_path, error_string_or_None).
    e.g. sanitize_path("../../etc/passwd") → error
    """
    resolved = os.path.realpath(os.path.abspath(os.path.expanduser(raw)))
    allowed  = os.path.realpath(allowed_root)
    if not resolved.startswith(allowed + os.sep) and resolved != allowed:
        return resolved, (
            f"Error: Path traversal blocked. "
            f"'{raw}' resolves to '{resolved}' which is outside '{allowed}'."
        )
    return resolved, None


def _assert_home(path: str, cmd: str) -> str | None:
    """
    Return an error string if `path` escapes the home directory (resolves symlinks).
    Returns None if the path is safe.
    """
    real_path = os.path.realpath(path)
    real_home = os.path.realpath(HOME_DIR)
    if not real_path.startswith(real_home + os.sep) and real_path != real_home:
        return (
            f"Error: '{cmd}' is restricted to your home directory. "
            f"Requested path '{path}' resolves to '{real_path}' which is outside '{real_home}'."
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

    # Expand ~ in string arguments that look like paths (not flags like -h)
    expanded_args = [
        _safe_path(a) if isinstance(a, str) and not a.startswith("-") else a
        for a in args
    ]

    # Home-directory restriction for sensitive read commands — blocks path traversal
    if base_cmd in HOME_RESTRICTED_CMDS and expanded_args:
        # Find the first non-flag argument (the target path)
        for i, arg in enumerate(expanded_args):
            if isinstance(arg, str) and not str(args[i]).startswith("-"):
                _, err = sanitize_path(arg, HOME_DIR)
                if err:
                    return err
                break

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


def search_files(pattern: str) -> str:
    """Search for files matching a glob pattern, starting from the home directory."""
    try:
        if not pattern:
            return "Error: Pattern cannot be empty."
        # Allow only safe glob characters (no shell injection)
        if re.search(r'[;|&`$<>]', pattern):
            return "Error: Pattern contains invalid characters."
        if os.path.isabs(pattern) or pattern.startswith("~"):
            search_pattern = os.path.expanduser(pattern)
        else:
            search_pattern = os.path.join(HOME_DIR, "**", pattern)
        matches = sorted(glob.glob(search_pattern, recursive=True))
        if not matches:
            return f"No files found matching '{pattern}'."
        result = "\n".join(matches[:100])
        if len(matches) > 100:
            result += f"\n... and {len(matches) - 100} more results."
        return result
    except Exception as e:
        logger.exception(f"search_files failed for '{pattern}': {e}")
        return f"Error searching files: {e}"


def find_text_in_files(text: str, search_dir: str = None) -> str:
    """Search for text within files using grep, restricted to the home directory."""
    try:
        if not text:
            return "Error: Search text cannot be empty."
        directory = _safe_path(search_dir) if search_dir else HOME_DIR
        err = _assert_home(directory, "find_text_in_files")
        if err:
            return err
        if not os.path.isdir(directory):
            return f"Error: Directory '{directory}' does not exist."
        rc, stdout, stderr = _run(
            ["grep", "-r", "-l", "-m", "1", "--", text, directory],
            timeout=30,
        )
        if rc == 0 and stdout:
            lines = stdout.splitlines()
            result = f"Text found in {len(lines)} file(s):\n" + "\n".join(lines[:50])
            if len(lines) > 50:
                result += f"\n... and {len(lines) - 50} more files."
            return result
        elif rc == 1:
            return f"No files containing '{text}' found in '{directory}'."
        else:
            return f"Error during search: {stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Search timed out."
    except FileNotFoundError:
        return "Error: 'grep' is not available on this system."
    except Exception as e:
        logger.exception(f"find_text_in_files failed: {e}")
        return f"Error searching files: {e}"


def list_processes() -> str:
    """List running processes for the current user, sorted by CPU usage."""
    try:
        current_user = os.getenv("USER") or os.getenv("LOGNAME") or ""
        procs = []
        for proc in psutil.process_iter(
            ["pid", "name", "username", "cpu_percent", "memory_percent", "status"]
        ):
            try:
                info = proc.info
                if not current_user or info.get("username") == current_user:
                    procs.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        procs.sort(key=lambda p: p.get("cpu_percent") or 0.0, reverse=True)
        header = f"{'PID':<8} {'NAME':<26} {'CPU%':>5}  {'MEM%':>5}  STATUS"
        sep = "-" * 60
        rows = [header, sep]
        for p in procs[:30]:
            rows.append(
                f"{p['pid']:<8} {(p['name'] or '')[:26]:<26} "
                f"{p.get('cpu_percent', 0):>5.1f}  "
                f"{p.get('memory_percent', 0):>5.1f}  "
                f"{p.get('status', 'unknown')}"
            )
        return "\n".join(rows)
    except Exception as e:
        logger.exception(f"list_processes failed: {e}")
        return f"Error listing processes: {e}"


def kill_process(pid: int) -> str:
    """Kill a process by PID. Only processes owned by the current user can be killed."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return f"Error: Invalid PID '{pid}'."
    try:
        proc = psutil.Process(pid)
        current_user = os.getenv("USER") or os.getenv("LOGNAME") or ""
        try:
            proc_user = proc.username()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return f"Error: Cannot access process {pid}."
        if current_user and proc_user != current_user:
            return f"Error: Cannot kill process {pid} owned by '{proc_user}'."
        proc_name = proc.name()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except psutil.TimeoutExpired:
            proc.kill()
        logger.info(f"Killed process {pid} ({proc_name})")
        return f"Process {pid} ({proc_name}) has been terminated."
    except psutil.NoSuchProcess:
        return f"Error: No process found with PID {pid}."
    except psutil.AccessDenied:
        return f"Error: Permission denied killing process {pid}."
    except Exception as e:
        logger.exception(f"kill_process failed for PID {pid}: {e}")
        return f"Error killing process {pid}: {e}"


def restart_process(name: str) -> str:
    """Restart a named systemd user service."""
    try:
        if not re.match(r'^[a-zA-Z0-9_.\-]+$', name):
            return f"Error: Invalid service name '{name}'."
        rc, stdout, stderr = _run(["systemctl", "--user", "restart", name], timeout=30)
        if rc == 0:
            return f"Service '{name}' restarted successfully."
        return (
            f"Error restarting '{name}': {stderr or 'unknown error'}. "
            "You may need sudo privileges or the service may not exist."
        )
    except FileNotFoundError:
        return "Error: 'systemctl' is not available on this system."
    except subprocess.TimeoutExpired:
        return f"Error: Restarting '{name}' timed out."
    except Exception as e:
        logger.exception(f"restart_process failed for '{name}': {e}")
        return f"Error restarting process '{name}': {e}"


def network_status() -> str:
    """Show network interfaces, addresses, and I/O statistics."""
    try:
        interfaces = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        io_counters = psutil.net_io_counters(pernic=True)
        lines = []
        family_names = {2: "IPv4", 10: "IPv6", 17: "MAC"}
        for iface, addrs in sorted(interfaces.items()):
            stat = stats.get(iface)
            status = "UP" if (stat and stat.isup) else "DOWN"
            speed = f"{stat.speed} Mbps" if (stat and stat.speed) else "unknown speed"
            lines.append(f"\n[{iface}] - {status} ({speed})")
            for addr in addrs:
                family = family_names.get(addr.family, str(addr.family))
                lines.append(f"  {family}: {addr.address}")
            if iface in io_counters:
                c = io_counters[iface]
                lines.append(
                    f"  Sent: {c.bytes_sent // 1024:,} KB  "
                    f"Recv: {c.bytes_recv // 1024:,} KB"
                )
        return "\n".join(lines) if lines else "No network interfaces found."
    except Exception as e:
        logger.exception(f"network_status failed: {e}")
        return f"Error getting network status: {e}"


def ping_host(host: str) -> str:
    """Ping a host 4 times and return the result."""
    try:
        if not re.match(r'^[a-zA-Z0-9.\-]+$', host):
            return f"Error: Invalid host '{host}'."
        rc, stdout, stderr = _run(["ping", "-c", "4", "-W", "3", host], timeout=20)
        return stdout if rc == 0 else f"Ping failed: {stderr or stdout}"
    except subprocess.TimeoutExpired:
        return "Error: Ping timed out."
    except FileNotFoundError:
        return "Error: 'ping' is not available on this system."
    except Exception as e:
        logger.exception(f"ping_host failed for '{host}': {e}")
        return f"Error pinging '{host}': {e}"


def traceroute_host(host: str) -> str:
    """Run traceroute (or tracepath) to a host."""
    try:
        if not re.match(r'^[a-zA-Z0-9.\-]+$', host):
            return f"Error: Invalid host '{host}'."
        for cmd_name in ("traceroute", "tracepath"):
            try:
                rc, stdout, stderr = _run([cmd_name, host], timeout=60)
                if rc == 0:
                    return stdout
                return f"{cmd_name} failed: {stderr or stdout}"
            except FileNotFoundError:
                continue
        return (
            "Error: Neither 'traceroute' nor 'tracepath' is available. "
            "Install with: sudo apt install traceroute"
        )
    except subprocess.TimeoutExpired:
        return "Error: Traceroute timed out."
    except Exception as e:
        logger.exception(f"traceroute_host failed for '{host}': {e}")
        return f"Error running traceroute to '{host}': {e}"


def download_file(url: str, dest: str) -> str:
    """Download a file from an HTTP/HTTPS/FTP URL to a local destination path."""
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https", "ftp"):
            return f"Error: URL scheme '{parsed.scheme}' is not allowed. Use http, https, or ftp."
        dest_path = _safe_path(dest)
        err = _assert_home(dest_path, "download_file")
        if err:
            return err
        dest_dir = os.path.dirname(dest_path)
        if dest_dir and not os.path.isdir(dest_dir):
            return f"Error: Destination directory '{dest_dir}' does not exist."
        rc, stdout, stderr = _run(
            ["curl", "-L", "--fail", "--output", dest_path, "--", url],
            timeout=300,
        )
        if rc == 0:
            return f"Successfully downloaded to '{dest_path}'."
        return f"Error downloading file: {stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Download timed out."
    except FileNotFoundError:
        return "Error: 'curl' is not installed. Install with: sudo apt install curl"
    except Exception as e:
        logger.exception(f"download_file failed for '{url}': {e}")
        return f"Error downloading '{url}': {e}"


def upload_file(filepath: str, destination: str) -> str:
    """
    Upload a file using rsync.
    Destination can be a local path or a remote path (user@host:/path).
    """
    try:
        src = _safe_path(filepath)
        if not os.path.exists(src):
            return f"Error: Source file '{src}' does not exist."
        # Block shell injection characters in destination
        if re.search(r'[;|&`$<>]', destination):
            return "Error: Invalid destination path."
        rc, stdout, stderr = _run(
            ["rsync", "-avz", "--progress", src, destination],
            timeout=300,
        )
        if rc == 0:
            return f"Successfully uploaded '{src}' to '{destination}'."
        return f"Error uploading file: {stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Upload timed out."
    except FileNotFoundError:
        return "Error: 'rsync' is not installed. Install with: sudo apt install rsync"
    except Exception as e:
        logger.exception(f"upload_file failed: {e}")
        return f"Error uploading '{filepath}': {e}"


def internet_speed() -> str:
    """Test internet download and upload speed using speedtest-cli."""
    try:
        import speedtest as _speedtest
        st = _speedtest.Speedtest()
        st.get_best_server()
        download_bps = st.download()
        upload_bps   = st.upload()
        ping_ms      = st.results.ping
        server       = st.results.server
        return (
            f"Ping:     {ping_ms:.1f} ms\n"
            f"Download: {download_bps / 1_000_000:.2f} Mbps\n"
            f"Upload:   {upload_bps   / 1_000_000:.2f} Mbps\n"
            f"Server:   {server.get('sponsor', 'unknown')} ({server.get('name', '')})"
        )
    except ImportError:
        return "Error: 'speedtest-cli' is not installed. Run ./setup.sh to install it."
    except Exception as e:
        logger.exception(f"internet_speed failed: {e}")
        return f"Error running speed test: {e}"


def _ollama_chat(prompt: str, system: str) -> str:
    """Internal helper: send a single prompt to Ollama and return the reply."""
    import requests as _requests
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        resp = _requests.get(f"{ollama_host}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        models = []
    if not models:
        return "Error: No Ollama models available. Make sure Ollama is running."
    payload = {
        "model": models[0],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    resp = _requests.post(f"{ollama_host}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "No response.")


def summarize_text(text: str) -> str:
    """Summarize input text using the local Ollama LLM."""
    try:
        if not text.strip():
            return "Error: Text cannot be empty."
        return _ollama_chat(
            f"Please summarize the following text concisely:\n\n{text}",
            "You are a helpful assistant that produces concise, accurate summaries.",
        )
    except Exception as e:
        logger.exception(f"summarize_text failed: {e}")
        return f"Error summarizing text: {e}"


def translate_text(text: str, lang: str) -> str:
    """Translate text to a target language using the local Ollama LLM."""
    try:
        if not text.strip():
            return "Error: Text cannot be empty."
        if not re.match(r'^[a-zA-Z\s]+$', lang):
            return f"Error: Invalid language '{lang}'. Use a language name like 'French'."
        return _ollama_chat(
            text,
            f"You are a professional translator. Translate the given text to {lang}. "
            "Return only the translation, without explanations.",
        )
    except Exception as e:
        logger.exception(f"translate_text failed: {e}")
        return f"Error translating text: {e}"


def convert_image(input_file: str, output_format: str) -> str:
    """Convert an image to another format using ImageMagick."""
    try:
        input_path = _safe_path(input_file)
        if not os.path.exists(input_path):
            return f"Error: Input file '{input_path}' does not exist."
        output_format = output_format.strip().lstrip(".").lower()
        if not re.match(r'^[a-z0-9]+$', output_format):
            return f"Error: Invalid output format '{output_format}'."
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}.{output_format}"
        if os.path.exists(output_path):
            return f"Error: Output file '{output_path}' already exists. Remove it first."
        rc, stdout, stderr = _run(["convert", input_path, output_path], timeout=60)
        if rc == 0:
            return f"Successfully converted to '{output_path}'."
        return f"Error converting image: {stderr}"
    except FileNotFoundError:
        return (
            "Error: 'convert' (ImageMagick) is not installed. "
            "Install with: sudo apt install imagemagick"
        )
    except Exception as e:
        logger.exception(f"convert_image failed: {e}")
        return f"Error converting image: {e}"


def resize_image(input_file: str, size: str) -> str:
    """Resize an image to a given size (e.g. '800x600') using ImageMagick."""
    try:
        input_path = _safe_path(input_file)
        if not os.path.exists(input_path):
            return f"Error: Input file '{input_path}' does not exist."
        if not re.match(r'^\d+x\d+$', size.strip()):
            return f"Error: Invalid size '{size}'. Use WxH format, e.g. '800x600'."
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_resized{ext}"
        rc, stdout, stderr = _run(
            ["convert", input_path, "-resize", size.strip(), output_path],
            timeout=60,
        )
        if rc == 0:
            return f"Successfully resized image saved to '{output_path}'."
        return f"Error resizing image: {stderr}"
    except FileNotFoundError:
        return (
            "Error: 'convert' (ImageMagick) is not installed. "
            "Install with: sudo apt install imagemagick"
        )
    except Exception as e:
        logger.exception(f"resize_image failed: {e}")
        return f"Error resizing image: {e}"


def analyze_image(input_file: str) -> str:
    """Analyze an image: return metadata via ImageMagick's identify command."""
    try:
        input_path = _safe_path(input_file)
        if not os.path.exists(input_path):
            return f"Error: Image file '{input_path}' does not exist."
        try:
            rc, stdout, _ = _run(["identify", "-verbose", input_path], timeout=15)
            if rc == 0 and stdout:
                key_fields = ("Geometry", "Type", "Colorspace", "Resolution", "Filesize", "Format")
                lines = [
                    line.strip()
                    for line in stdout.splitlines()
                    if any(f in line for f in key_fields)
                ]
                return "Image metadata:\n" + "\n".join(lines[:15]) if lines else stdout[:500]
        except FileNotFoundError:
            pass
        # Fallback: basic file info
        size = os.path.getsize(input_path)
        return f"File: {os.path.basename(input_path)}, Size: {size:,} bytes"
    except Exception as e:
        logger.exception(f"analyze_image failed: {e}")
        return f"Error analyzing image: {e}"


# Commands allowed for task scheduling
_SCHEDULE_ALLOWED = WHITELISTED_CMDS | {"backup", "sync", "cleanup", "update"}


def schedule_task(command: str, time: str) -> str:
    """
    Schedule a whitelisted command at a specific time using the 'at' daemon.
    Time format examples: '14:30', '02:00 tomorrow', '10:00 2026-04-01'.
    """
    try:
        if not time.strip():
            return "Error: Time cannot be empty."
        if re.search(r'[;|&`$<>]', time):
            return f"Error: Invalid time string '{time}'."
        base_cmd = command.strip().split()[0]
        if base_cmd not in _SCHEDULE_ALLOWED:
            return (
                f"Error: Command '{base_cmd}' is not allowed for scheduling. "
                f"Allowed: {', '.join(sorted(_SCHEDULE_ALLOWED))}"
            )
        proc = subprocess.run(
            ["at", time.strip()],
            input=command,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0 or "job" in proc.stderr.lower():
            return f"Task scheduled: '{command}' at '{time}'. {proc.stderr.strip()}"
        return f"Error scheduling task: {proc.stderr}"
    except FileNotFoundError:
        return "Error: 'at' is not installed. Install with: sudo apt install at"
    except subprocess.TimeoutExpired:
        return "Error: Task scheduling timed out."
    except Exception as e:
        logger.exception(f"schedule_task failed: {e}")
        return f"Error scheduling task: {e}"


def set_reminder(message: str, time: str) -> str:
    """
    Set a terminal reminder at a specific time using the 'at' daemon.
    Time format examples: '14:30', '09:00 tomorrow'.
    """
    try:
        if not time.strip():
            return "Error: Time cannot be empty."
        if re.search(r'[;|&`$<>]', time):
            return f"Error: Invalid time string '{time}'."
        # Sanitise message to prevent shell injection
        safe_message = re.sub(r"[^a-zA-Z0-9 .,!?:;@\-_'\"()]", "", message)
        if not safe_message:
            return "Error: Reminder message is empty after sanitization."
        tty = os.ttyname(1) if os.isatty(1) else "/dev/console"
        echo_cmd = f'echo "REMINDER: {safe_message}" > {tty}'
        proc = subprocess.run(
            ["at", time.strip()],
            input=echo_cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0 or "job" in proc.stderr.lower():
            return f"Reminder set for '{time}': '{safe_message}'. {proc.stderr.strip()}"
        return f"Error setting reminder: {proc.stderr}"
    except FileNotFoundError:
        return "Error: 'at' is not installed. Install with: sudo apt install at"
    except subprocess.TimeoutExpired:
        return "Error: Setting reminder timed out."
    except Exception as e:
        logger.exception(f"set_reminder failed: {e}")
        return f"Error setting reminder: {e}"


# ─────────────────────────────────────────────
# TOOL REGISTRY
# ─────────────────────────────────────────────

AVAILABLE_TOOLS = {
    # System
    "get_system_status":  get_system_status,
    "gpu_status":         gpu_status,
    "check_updates":      check_updates,
    "run_safe_command":   run_safe_command,
    "list_directory":     list_directory,
    "open_file":          open_file,
    # File operations
    "search_files":       search_files,
    "find_text_in_files": find_text_in_files,
    # Process management
    "list_processes":     list_processes,
    "kill_process":       kill_process,
    "restart_process":    restart_process,
    # Network
    "network_status":     network_status,
    "ping_host":          ping_host,
    "traceroute_host":    traceroute_host,
    "internet_speed":     internet_speed,
    # File transfer
    "download_file":      download_file,
    "upload_file":        upload_file,
    # Media
    "download_youtube":   download_youtube,
    "convert_video":      convert_video,
    "convert_image":      convert_image,
    "resize_image":       resize_image,
    "analyze_image":      analyze_image,
    # LLM utilities
    "summarize_text":     summarize_text,
    "translate_text":     translate_text,
    # Scheduler
    "schedule_task":      schedule_task,
    "set_reminder":       set_reminder,
}