import subprocess
import psutil
import os

def list_directory(path=".", show_sizes=False, include_dir_size=False):
    """List files and folders with optional size info (read-only, safe)."""
    try:
        # Expand ~ if it exists and make absolute
        path = os.path.abspath(os.path.expanduser(path))
        items = os.listdir(path)
        result = []

        for item in items:
            full_path = os.path.join(path, item)

            if os.path.isdir(full_path):
                if show_sizes and include_dir_size:
                    size = get_dir_size(full_path)
                    result.append(f"[DIR] {item} ({size} bytes)")
                else:
                    result.append(f"[DIR] {item}")
            else:
                if show_sizes:
                    size = os.path.getsize(full_path)
                    result.append(f"[FILE] {item} ({size} bytes)")
                else:
                    result.append(f"[FILE] {item}")

        return "\n".join(result) if result else "Directory is empty."

    except Exception as e:
        return f"Error: {str(e)}"

def get_dir_size(path):
    """Calculate total size of a directory (recursive)."""
    path = os.path.expanduser(path)
    total_size = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)
            except:
                pass
    return total_size

def run_safe_command(base_cmd, *args):
    """Run a whitelisted base command with optional arguments."""
    WHITELISTED_BASES = {
        "nvidia-smi", "whoami", "uptime", "df", "ls", "cat", "yt-dlp", "ffmpeg", "ffprobe"
    }
    
    if base_cmd not in WHITELISTED_BASES:
        return f"Error: Command '{base_cmd}' is not in the safe whitelist. Available: {', '.join(sorted(WHITELISTED_BASES))}"
    
    try:
        # Construct the command list: [base_cmd, arg1, arg2, ...]
        cmd = [base_cmd] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout if result.stdout else "Command executed successfully (no output)."
        else:
            return f"Error executing {base_cmd}: {result.stderr}"
    except Exception as e:
        return f"Error executing command: {e}"

def gpu_status():
    """Shortcut for nvidia-smi."""
    return run_safe_command("nvidia-smi")

def get_system_status():
    """Get CPU, RAM usage and top processes."""
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        
        # Get top 3 processes by memory usage
        processes = []
        for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        top_cpu = sorted(processes, key=lambda p: p['cpu_percent'], reverse=True)[:3]
        top_mem = sorted(processes, key=lambda p: p['memory_percent'], reverse=True)[:3]
        
        status = {
            "cpu_percent": cpu_usage,
            "ram_total_gb": round(ram.total / (1024**3), 2),
            "ram_used_gb": round(ram.used / (1024**3), 2),
            "ram_percent": ram.percent,
            "top_processes_cpu": top_cpu,
            "top_processes_mem": top_mem
        }
        return str(status)
    except Exception as e:
        return f"Error getting system status: {e}"

def check_updates():
    """Check for system updates on Linux (Debian/Ubuntu)."""
    try:
        # Run apt list --upgradable
        result = subprocess.run(['apt', 'list', '--upgradable'], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        
        # The first line is usually "Listing..."
        upgradable = [line for line in lines if '/' in line]
        count = len(upgradable)
        
        if count == 0:
            return "Your system is up to date."
        else:
            summary = f"Found {count} upgradable packages."
            if count > 0:
                examples = ", ".join([pkg.split('/')[0] for pkg in upgradable[:5]])
                summary += f" Examples: {examples}"
                if count > 5:
                    summary += " ..."
            return summary
    except Exception as e:
        return f"Error checking updates: {e}"

def download_youtube(url):
    """Download a YouTube video using yt-dlp."""
    try:
        # We use -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' to ensure mp4 format
        # and --no-playlist to avoid downloading entire playlists
        cmd = ['yt-dlp', '-f', 'mp4', '--no-playlist', '-o', '%(title)s.%(ext)s', url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return f"Successfully downloaded video from {url}"
        else:
            return f"Error downloading video: {result.stderr}"
    except Exception as e:
        return f"Error executing yt-dlp: {e}"

def convert_video(input_file, output_format):
    """Convert a video file to another format using ffmpeg."""
    try:
        output_file = input_file.rsplit('.', 1)[0] + '.' + output_format
        cmd = ['ffmpeg', '-i', input_file, output_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return f"Successfully converted {input_file} to {output_file}"
        else:
            return f"Error converting video: {result.stderr}"
    except Exception as e:
        return f"Error executing ffmpeg: {e}"

# Tool Registry
AVAILABLE_TOOLS = {
    "check_updates": check_updates,
    "download_youtube": download_youtube,
    "convert_video": convert_video,
    "get_system_status": get_system_status,
    "run_safe_command": run_safe_command,
    "gpu_status": gpu_status,
    "list_directory": list_directory
}
