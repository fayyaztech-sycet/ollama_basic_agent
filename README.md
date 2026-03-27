# Local Ollama Agent

A modular, tool-enabled AI agent that runs locally on Linux using Ollama.

## Features

- **Local Inference**: Uses Ollama for LLM processing.
- **Streaming Responses**: Real-time token output in the terminal.
- **Tool Integration**:
    - **System Status**: Monitor CPU, RAM, and top processes.
    - **Media Management**: Download YouTube videos (`yt-dlp`) and convert formats (`ffmpeg`).
    - **Safe Command Runner**: Execute whitelisted Linux commands (`ls`, `cat`, `df`, `nvidia-smi`, etc.).
    - **Update Checker**: Check for system updates via `apt`.
- **Strict Execution**: The agent only operates through its authorized tools.

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

## Usage

Start the agent with:
```bash
./run.sh
```

Select your preferred model from the list and start chatting!

## Tools Registry

- `get_system_status()`: Real-time resource monitoring.
- `gpu_status()`: Quick access to `nvidia-smi`.
- `download_youtube(url)`: High-quality video downloads.
- `convert_video(input, format)`: Flexible file conversion.
- `run_safe_command(base, *args)`: Generalized whitelisted command runner.
- `check_updates()`: System upgrade check.
