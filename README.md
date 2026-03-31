---

## Token Usage Dashboard

Track your daily AI token usage with a built-in dashboard. The dashboard records and displays the number of tokens processed by the agent each day, helping you monitor usage trends and optimize performance.

- **Features:**
   - Daily token count and summary
   - Historical usage trends
   - Optional alerts for high usage

**Example:**

| Date       | Tokens Used |
|------------|-------------|
| 2026-03-29 |  12,345     |
| 2026-03-30 |  10,210     |

The dashboard can be accessed via a command or web interface (to be implemented).

---

## Token Usage Dashboard Usage

- **Log token usage after each request:**
  ```python
  from tools import log_token_usage
  log_token_usage(input_tokens, output_tokens)
  ```
- **Show dashboard for last 7 days:**
  ```python
  from tools import get_token_dashboard
  print(get_token_dashboard(7))
  ```

Replace `input_tokens` and `output_tokens` with the number of tokens processed in your request and response.

The dashboard now also shows estimated cost per day (USD), based on the rates set in `tools.py`.

---

## Local Ollama Agent

A modular, tool-enabled AI agent that runs locally on Linux using Ollama.

---

## Features

- **Local Inference**: Uses Ollama for LLM processing.
- **Streaming Responses**: Real-time token output in the terminal.
- **Tool Integration**:
      - **System Status**: Monitor CPU, RAM, and top processes.
      - **Media Management**: Download YouTube videos (`yt-dlp`) and convert formats (`ffmpeg`).
      - **Safe Command Runner**: Execute whitelisted Linux commands (`ls`, `cat`, `df`, `nvidia-smi`, etc.).
      - **Update Checker**: Check for system updates via `apt`.
- **Strict Execution**: The agent only operates through its authorized tools.

---

## Requirements

- Linux OS
- Python 3.8+
- [Ollama](https://ollama.com) installed and running
- `yt-dlp`, `ffmpeg`, `nvidia-smi`, and `apt` available in PATH
- Python packages: see `setup.sh` or `requirements.txt` (if available)

---

## Setup

1. **Install Ollama**: Follow the instructions at [ollama.com](https://ollama.com).
2. **Download Models**:
    ```bash
    ollama pull qwen2.5-coder:7b
    ```
3. **Run Setup Script**:
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

---

## Usage

Start the agent:
```bash
./run.sh
```

Select your preferred model from the list and start chatting!

### Example Interactions

- **Check system status:**
   > get_system_status()
- **Download a YouTube video:**
   > download_youtube("https://youtube.com/...")
- **Convert a video:**
   > convert_video("input.mp4", "avi")
- **Run a safe command:**
   > run_safe_command("ls", "-l")

---

## Tools Registry

| Tool | Description | Example |
|------|-------------|---------|
| `get_system_status()` | Real-time resource monitoring | `get_system_status()` |
| `gpu_status()` | Quick access to `nvidia-smi` | `gpu_status()` |
| `download_youtube(url)` | High-quality video downloads | `download_youtube("<url>")` |
| `convert_video(input, format)` | Flexible file conversion | `convert_video("input.mp4", "avi")` |
| `run_safe_command(base, *args)` | Generalized whitelisted command runner | `run_safe_command("ls", "-l")` |
| `check_updates()` | System upgrade check | `check_updates()` |
| `search_files(pattern)` | Search for files matching a pattern | `search_files("*.mp4")` |
| `find_text_in_files(text)` | Search for text within files | `find_text_in_files("import os")` |
| `list_processes()` | List running processes | `list_processes()` |
| `kill_process(pid)` | Kill a process by PID | `kill_process(1234)` |
| `restart_process(name)` | Restart a named process | `restart_process("nginx")` |
| `network_status()` | Show network interfaces and status | `network_status()` |
| `ping_host(host)` | Ping a host | `ping_host("8.8.8.8")` |
| `traceroute_host(host)` | Run traceroute to a host | `traceroute_host("example.com")` |
| `download_file(url, dest)` | Download a file from a URL | `download_file("https://example.com/file.txt", "./file.txt")` |
| `upload_file(filepath, destination)` | Upload a file to a cloud or remote location | `upload_file("./file.txt", "cloud:/folder/")` |
| `summarize_text(text)` | Summarize input text using LLM | `summarize_text("Long article text...")` |
| `translate_text(text, lang)` | Translate text to a target language | `translate_text("Hello", "fr")` |
| `convert_image(input, format)` | Convert image format | `convert_image("img.png", "jpg")` |
| `resize_image(input, size)` | Resize image to given size | `resize_image("img.jpg", "800x600")` |
| `analyze_image(input)` | Analyze image content | `analyze_image("img.jpg")` |
| `schedule_task(command, time)` | Schedule a command/task | `schedule_task("backup.sh", "02:00")` |
| `set_reminder(message, time)` | Set a reminder | `set_reminder("Meeting at 3pm", "15:00")` |

---

## Tool Implementation Status

| Tool | Status |
|------|--------|
| get_system_status | ✅ Completed |
| gpu_status | ✅ Completed |
| download_youtube | ✅ Completed |
| convert_video | ✅ Completed |
| run_safe_command | ✅ Completed |
| check_updates | ✅ Completed |
| search_files | ✅ Completed |
| find_text_in_files | ✅ Completed |
| list_processes | ✅ Completed |
| kill_process | ✅ Completed |
| restart_process | ✅ Completed |
| network_status | ✅ Completed |
| ping_host | ✅ Completed |
| traceroute_host | ✅ Completed |
| download_file | ✅ Completed |
| upload_file | ✅ Completed |
| summarize_text | ✅ Completed |
| translate_text | ✅ Completed |
| convert_image | ✅ Completed |
| resize_image | ✅ Completed |
| analyze_image | ✅ Completed |
| schedule_task | ✅ Completed |
| set_reminder | ✅ Completed |

---

## Additional Utility Ideas

- **File Search:** Search for files or text within files.
- **Process Management:** List, kill, or restart processes.
- **Network Tools:** Check network status, ping, or traceroute.
- **File Upload/Download:** Upload files to cloud or download from URLs.
- **Text Summarization/Translation:** Use LLM for summarizing or translating text.
- **Image Tools:** Convert, resize, or analyze images.
- **Scheduler:** Schedule tasks or reminders.

---

## Troubleshooting

- **Ollama not found:** Ensure Ollama is installed and running.
- **Missing dependencies:** Run `./setup.sh` to install required packages.
- **Permission denied:** Use `chmod +x run.sh setup.sh` to make scripts executable.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for new features, bug fixes, or improvements.

---

## License

Specify your license here (e.g., MIT, Apache 2.0).
