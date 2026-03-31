import sys
import os
import re
import json
import logging
import argparse
from datetime import datetime
from dotenv import load_dotenv
from ollama_service import OllamaService
from tools import AVAILABLE_TOOLS

# Load .env file if present
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ─────────────────────────────────────────────
# ANSI COLORS
# ─────────────────────────────────────────────
COLOR_USER  = "\033[96m"   # Cyan  — user input
COLOR_AI    = "\033[92m"   # Green — AI response
COLOR_DEBUG = "\033[33m"   # Yellow — debug steps
COLOR_RESET = "\033[0m"

# DEBUG_MODE is set after argparse in main(); default False until then
DEBUG_MODE = False

# ─────────────────────────────────────────────
# CONFIG LOADER
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    "ollama_host":     os.getenv("OLLAMA_HOST",     "http://localhost:11434"),
    "max_steps":       int(os.getenv("MAX_STEPS",    "5")),
    "history_window":  int(os.getenv("HISTORY_WINDOW", "14")),
    "log_file":        os.getenv("LOG_FILE",        "agent.log"),
    "log_level":       os.getenv("LOG_LEVEL",       "INFO"),
    "memory_file":     os.getenv("MEMORY_FILE",     "memory.json"),
}

def load_config(path="settings.json") -> dict:
    """Load settings.json or create it with defaults if missing."""
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                user_config = json.load(f)
                merged = {**DEFAULT_CONFIG, **user_config}
                return _validate_config(merged)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[Warning] Could not read '{path}': {e}. Using defaults.")
    else:
        # First run: write defaults so user can edit
        try:
            with open(path, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            print(f"[Info] Created default '{path}'. Edit it to customize the agent.")
        except IOError as e:
            print(f"[Warning] Could not create '{path}': {e}.")
    return DEFAULT_CONFIG.copy()


_CONFIG_TYPES = {
    "ollama_host":    str,
    "max_steps":      int,
    "history_window": int,
    "log_file":       str,
    "log_level":      str,
    "memory_file":    str,
}

def _validate_config(cfg: dict) -> dict:
    """Coerce config values to expected types; fall back to defaults on failure."""
    out = dict(cfg)
    for key, typ in _CONFIG_TYPES.items():
        if key not in out:
            out[key] = DEFAULT_CONFIG[key]
            continue
        try:
            out[key] = typ(out[key])
        except (ValueError, TypeError):
            print(f"[Warning] Config key '{key}' has invalid value '{out[key]}'; "
                  f"using default '{DEFAULT_CONFIG[key]}'.")
            out[key] = DEFAULT_CONFIG[key]
    return out


# ─────────────────────────────────────────────
# LOGGER SETUP
# ─────────────────────────────────────────────

def setup_logger(log_file: str, log_level: str, debug_mode: bool = False) -> logging.Logger:
    """Configure a logger that writes to both file and stdout."""
    logger = logging.getLogger("agent")
    level = logging.DEBUG if debug_mode else getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid adding duplicate handlers on re-import
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler — always full detail
    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except IOError as e:
        print(f"[Warning] Cannot write to log file '{log_file}': {e}")

    # Console handler — DEBUG when --debug, else WARNING (keeps terminal clean)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG if debug_mode else logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)

    return logger


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """You are a helpful Linux System Agent with persistent memory. You MUST respond in valid JSON format ONLY.

RESPONSE FORMAT:
{{
  "message": "What to display to the user. See rules below.",
  "tool": "tool_name or null",
  "args": ["arg1", "arg2"]
}}

CRITICAL RULE — "message" field:
- When "tool" is null  → "message" = your complete reply, written directly to the user as natural speech.
  BAD:  "User greeted me. I will now list my capabilities and set tool to null."
  GOOD: "Hello! I'm your local Linux AI assistant. Here's what I can do: ..."
- When "tool" is set   → "message" = one short sentence explaining why you chose that tool.

STRICT RULES:
1. Respond with the JSON object ONLY. Double-quoted keys and values. No markdown, no code fences, no extra text before or after the JSON.
2. NEVER guess or make up system facts. For ANY of the following, you MUST call a tool — never answer from memory:
   - IP address, hostname, username, current time/date, disk space, CPU/RAM, GPU, network interfaces, running processes, system uptime, OS info.
3. For greetings or purely conversational questions: set "tool": null, write the reply in "message".
4. NEVER provide manual terminal commands for the user to run.
5. "args" MUST always be a JSON array, even when empty: [].
6. FLAGS vs PATHS: Flags like -h, -l, -a must stay as flags (e.g. ["df", "-h"]). Never expand a flag into a path.
7. CONTEXT: If the answer is already in a previous tool result in the conversation, answer directly without calling the tool again.
8. "is X working" / "is X reachable" / "check X" for a hostname → use ping_host(X).

AVAILABLE TOOLS:
System:
- get_system_status(): Returns CPU/RAM usage and top processes.
- gpu_status(): Returns GPU status via nvidia-smi.
- check_updates(): Checks for system updates via apt.
- run_safe_command(base_cmd, *args): Runs a whitelisted command (ls, cat, df, uptime, date, hostname, uname, nvidia-smi, yt-dlp, ffmpeg, ffprobe, whoami).
- list_directory(path, show_sizes, include_dir_size): Lists files/folders.
- open_file(path): Opens a file or directory with the default handler.

File Operations:
- search_files(pattern): Search for files matching a glob pattern in home directory.
- find_text_in_files(text, search_dir): Search for text inside files using grep.

Process Management:
- list_processes(): List running processes for the current user.
- kill_process(pid): Kill a process by PID (own processes only).
- restart_process(name): Restart a named systemd user service.

Network:
- network_status(): Show network interfaces and I/O stats.
- ping_host(host): Ping a host 4 times.
- traceroute_host(host): Run traceroute to a host.
- internet_speed(): Test internet download/upload speed and ping.

File Transfer:
- download_file(url, dest): Download a file from an HTTP/HTTPS/FTP URL (curl).
- upload_file(filepath, destination): Upload a file via rsync.

Media:
- download_youtube(url): Download a YouTube video in best quality.
- convert_video(input_file, output_format): Convert video format with ffmpeg.
- convert_image(input_file, output_format): Convert image format with ImageMagick.
- resize_image(input_file, size): Resize an image (e.g. "800x600") with ImageMagick.
- analyze_image(input_file): Show image metadata via ImageMagick identify.

LLM Utilities:
- summarize_text(text): Summarize text using the local Ollama model.
- translate_text(text, lang): Translate text to a target language.

Scheduler:
- schedule_task(command, time): Schedule a whitelisted command via the 'at' daemon.
- set_reminder(message, time): Set a terminal reminder via the 'at' daemon.

TOOL USAGE EXAMPLES:
- User: "hi" / "hello"           → {{"message": "Hello! I'm your local Linux AI assistant powered by Ollama. Here's what I can help you with:\n\n• System monitoring — CPU, RAM, GPU, disk, processes\n• File operations — search, list, read files\n• Network tools — ping, traceroute, interface status, speed test\n• Media — download YouTube, convert/resize/analyze images & videos\n• Process management — list, kill, restart processes\n• File transfer — download from URLs, upload via rsync\n• AI utilities — summarize or translate text\n• Scheduler — schedule tasks and set reminders\n\nWhat would you like to do?", "tool": null, "args": []}}
- User: "what is my IP" / "local IP" → {{"message": "Fetching network interfaces.", "tool": "network_status", "args": []}}
- User: "what is my username"    → {{"message": "Getting username.", "tool": "run_safe_command", "args": ["whoami"]}}
- User: "what time is it"        → {{"message": "Checking current time.", "tool": "run_safe_command", "args": ["date"]}}
- User: "what's my CPU/RAM?"     → {{"message": "Fetching system status.", "tool": "get_system_status", "args": []}}
- User: "ping google.com"        → {{"message": "Pinging google.com.", "tool": "ping_host", "args": ["google.com"]}}
- User: "is google.com working"  → {{"message": "Checking if google.com is reachable.", "tool": "ping_host", "args": ["google.com"]}}
- User: "check if 8.8.8.8 is up" → {{"message": "Pinging 8.8.8.8.", "tool": "ping_host", "args": ["8.8.8.8"]}}
- User: "read ~/.bashrc"         → {{"message": "Reading .bashrc.", "tool": "run_safe_command", "args": ["cat", "~/.bashrc"]}}
- User: "list home directory"    → {{"message": "Listing home directory.", "tool": "run_safe_command", "args": ["ls", "~"]}}
- User: "check disk space"       → {{"message": "Checking disk space.", "tool": "run_safe_command", "args": ["df", "-h"]}}
- User: "find *.mp4 files"       → {{"message": "Searching for mp4 files.", "tool": "search_files", "args": ["*.mp4"]}}
- User: "list processes"         → {{"message": "Listing running processes.", "tool": "list_processes", "args": []}}
- User: "network status"         → {{"message": "Checking network interfaces.", "tool": "network_status", "args": []}}
- User: "test internet speed"    → {{"message": "Running speed test.", "tool": "internet_speed", "args": []}}

TIPS:
- Paths: "~" expands to your home directory.
- Memory: Recap what you did recently.
- Truth: Trust tool output. If a tool lists a file, IT IS THERE.

REASONING STEPS:
1. If the user asks for a file (e.g., "firebase video"):
   a. Call list_directory(parent_path) to see what's actually there.
   b. Look at the result. If you see "firebase integration.mp4", THAT IS THE FILE.
   c. Use the EXACT name from tool output for your next action.
2. NEVER say a file is missing if it appeared in tool output.
3. If no match is found, apologize and ask for the correct name.

MEDIA REASONING:
- If asked to "check" or "show formats": Use run_safe_command("yt-dlp", "-F", url).
- NEVER call download_youtube(url) unless "download" or "get" is clearly intended.

SSH KEY SECURITY:
- Files ending in .pub (e.g. id_rsa.pub, id_ed25519.pub) are PUBLIC keys — safe to read and display freely.
- Files WITHOUT .pub (e.g. id_rsa, id_ed25519) are PRIVATE keys — NEVER read or display them under any circumstance.
- If asked to read a .pub file, use run_safe_command("cat", "~/.ssh/<filename>.pub").

CURRENT MEMORY:
{memory_context}
"""


# ─────────────────────────────────────────────
# MEMORY MANAGER
# ─────────────────────────────────────────────

class MemoryManager:
    def __init__(self, filename: str, logger: logging.Logger):
        self.filename = filename
        self.logger = logger
        self.memory = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r") as f:
                    data = json.load(f)
                    self.logger and self.logger.debug(f"Memory loaded from '{self.filename}'.")
                    return data
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Could not load memory from '{self.filename}': {e}. Starting fresh.")
        return {"last_file": "None", "last_command": "None", "notes": [], "interactions": []}

    def save(self):
        try:
            with open(self.filename, "w") as f:
                json.dump(self.memory, f, indent=2)
        except IOError as e:
            self.logger.warning(f"Could not save memory: {e}")

    def add_interaction(self, user_msg: str, agent_reply: str, max_entries: int = 20):
        """Store a rolling summary of recent user↔agent exchanges."""
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "user": user_msg[:120],
            "agent": agent_reply[:200],
        }
        interactions = self.memory.setdefault("interactions", [])
        interactions.append(entry)
        # Keep only the last max_entries
        if len(interactions) > max_entries:
            self.memory["interactions"] = interactions[-max_entries:]
        self.save()

    def update(self, tool_name: str, result: str, args: list):
        self.memory["last_command"] = f"{tool_name}({', '.join(map(str, args))})"

        if tool_name == "download_youtube":
            # yt-dlp prints the saved filepath on its last stdout line
            last_line = result.strip().splitlines()[-1] if result.strip() else ""
            if last_line and os.path.exists(os.path.expanduser(last_line)):
                self.memory["last_file"] = last_line
            elif "as " in result:
                match = re.search(r"as (.+)$", result, re.MULTILINE)
                if match:
                    self.memory["last_file"] = match.group(1).strip()

        elif tool_name == "convert_video" and "Successfully converted" in result:
            match = re.search(r"to (.+)$", result)
            if match:
                self.memory["last_file"] = match.group(1).strip()

        self.save()
        self.logger.debug(f"Memory updated: {self.memory}")

    def get_context(self) -> str:
        ctx = {
            "last_file":    self.memory.get("last_file", "None"),
            "last_command": self.memory.get("last_command", "None"),
            "notes":        self.memory.get("notes", []),
            "recent_interactions": self.memory.get("interactions", [])[-5:],
        }
        return json.dumps(ctx, indent=2)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def extract_json(text: str) -> dict | None:
    """
    Extract and parse a JSON object from LLM output.

    Handles:
    - Markdown code fences (```json ... ```)
    - Prose before/after the JSON object
    - Single-quoted keys/values (simple cases)
    - Both 'message' (new) and 'thought' (legacy) field names
    """
    # 1. Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    cleaned = cleaned.replace("```", "").strip()

    # 2. Discard anything before the first '{' (prose/headers the model prepends)
    brace_start = cleaned.find('{')
    if brace_start > 0:
        cleaned = cleaned[brace_start:]

    def _try_parse(s: str) -> dict | None:
        """Try JSON parse; if it fails, retry after converting single quotes."""
        for candidate in (s, s.replace("'", '"')):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and ("message" in obj or "thought" in obj):
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    # 3. Greedy scan: find the largest {...} block spanning the whole object.
    #    Walk from each '{' forward, tracking brace depth.
    best: dict | None = None
    best_len = 0
    for start in range(len(cleaned)):
        if cleaned[start] != '{':
            continue
        depth = 0
        in_string = False
        escape = False
        for end in range(start, len(cleaned)):
            ch = cleaned[end]
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    span = cleaned[start:end + 1]
                    if len(span) > best_len:
                        obj = _try_parse(span)
                        if obj:
                            best = obj
                            best_len = len(span)
                    break
    if best:
        return best

    # 3. Last resort: the whole cleaned text
    return _try_parse(cleaned.strip())


def select_model(models: list, logger: logging.Logger) -> str:
    """Interactive model selection with input validation."""
    print("\nAvailable Models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")

    while True:
        choice = input("\nSelect a model (number) or press Enter for the first one: ").strip()
        if not choice:
            logger.info(f"No selection made — defaulting to '{models[0]}'.")
            return models[0]
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                logger.info(f"Model selected: '{models[idx]}'.")
                return models[idx]
        print(f"  [!] Invalid choice. Please enter a number between 1 and {len(models)}.")


# ─────────────────────────────────────────────
# MAIN AGENT LOOP
# ─────────────────────────────────────────────

_RETRY_MSG = (
    'INVALID RESPONSE. Reply with ONLY a raw JSON object — '
    'no markdown, no code fences, no prose before or after the braces. '
    'Required structure: {"message": "<text>", "tool": null, "args": []} '
    'or {"message": "<reason>", "tool": "<tool_name>", "args": ["<arg1>"]}'
)


def _call_llm_with_validation(
    service: OllamaService,
    model: str,
    messages: list,
    logger: logging.Logger,
    max_retries: int = 3,
) -> tuple[dict | None, str]:
    """
    Call the LLM and retry up to max_retries times when JSON is invalid.
    Returns (parsed_data, raw_response). Streams tokens live to terminal.
    """
    msgs = list(messages)   # local copy so we can append retry hints
    raw = ""

    for attempt in range(max_retries):
        raw = ""
        stream_error = False

        # ── Live streaming: print tokens as they arrive ──
        print(f"\n{COLOR_AI}Agent:{COLOR_RESET} ", end="", flush=True)
        try:
            for chunk in service.chat(model, msgs, stream=True):
                raw += chunk
                # Only print if it looks like prose (not JSON scaffolding noise)
                print(chunk, end="", flush=True)
        except RuntimeError as e:
            logger.error(f"Stream error (attempt {attempt + 1}): {e}")
            stream_error = True

        print()   # newline after streamed content

        if stream_error:
            break

        data = extract_json(raw)
        if data:
            return data, raw

        logger.warning(
            f"Invalid JSON on attempt {attempt + 1}/{max_retries}. "
            f"Raw (first 200): {raw[:200]!r}"
        )
        msgs.append({"role": "user", "content": _RETRY_MSG})

    return None, raw


def run_agent(config: dict, logger: logging.Logger):
    service = OllamaService(base_url=config["ollama_host"])
    mem = MemoryManager(filename=config["memory_file"], logger=logger)

    # ── Startup connectivity check ──
    if not service.is_available():
        logger.error(
            f"Cannot connect to Ollama at '{config['ollama_host']}'. "
            "Make sure Ollama is running (`ollama serve`)."
        )
        sys.exit(1)

    logger.info("Checking for available Ollama models...")
    models = service.list_models()

    if not models:
        logger.error(
            "No models found in Ollama. "
            "Pull one first with: ollama pull <model_name>"
        )
        sys.exit(1)

    selected_model = select_model(models, logger)
    print(f"\nUsing model : {selected_model}")
    print(f"Log file    : {config['log_file']}")
    print(f"Memory file : {config['memory_file']}")
    print(f"Debug mode  : {'ON' if DEBUG_MODE else 'OFF'}")
    print(f"\n{COLOR_AI}Agent is ready!{COLOR_RESET} (type 'quit' or 'exit' to stop)\n")
    print("─" * 50)

    MAX_STEPS      = config["max_steps"]
    HISTORY_WINDOW = config["history_window"]
    # Ensure we always slice on an even boundary so user/assistant pairs stay together
    if HISTORY_WINDOW % 2 != 0:
        HISTORY_WINDOW += 1
    history: list[dict] = []

    while True:
        try:
            user_input = input(f"\n{COLOR_USER}You:{COLOR_RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{COLOR_AI}[Agent] Goodbye!{COLOR_RESET}")
            logger.info("Session ended by user (KeyboardInterrupt / EOF).")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print(f"{COLOR_AI}[Agent] Goodbye!{COLOR_RESET}")
            logger.info("Session ended by user command.")
            break

        logger.debug(f"User: {user_input}")
        history.append({"role": "user", "content": user_input})

        current_system_prompt = BASE_SYSTEM_PROMPT.format(
            memory_context=mem.get_context()
        )

        # Slice in pairs so no user message is orphaned without its assistant reply
        window = history[-(HISTORY_WINDOW):]
        if window and window[0]["role"] == "assistant":
            window = window[1:]

        messages = [{"role": "system", "content": current_system_prompt}] + window

        last_tool_call = None
        responded      = False

        for step in range(1, MAX_STEPS + 1):
            if DEBUG_MODE:
                print(f"{COLOR_DEBUG}[Step {step}/{MAX_STEPS}]{COLOR_RESET} ", end="", flush=True)
            logger.debug(f"Step {step}: sending {len(messages)} messages to model.")

            data, raw = _call_llm_with_validation(
                service, selected_model, messages, logger
            )

            if not data:
                logger.error(f"Step {step}: could not parse JSON after retries. Raw: {raw[:200]}")
                print(f"({COLOR_DEBUG}could not parse a valid response — please rephrase{COLOR_RESET})")
                break

            # Support both 'message' (new) and 'thought' (legacy)
            message   = data.get("message") or data.get("thought", "")
            tool_name = data.get("tool")
            args      = data.get("args", [])

            if not isinstance(args, list):
                args = [args]

            logger.debug(f"Step {step} | tool={tool_name} | args={args} | msg={message[:80]}")
            if DEBUG_MODE:
                print(f"{COLOR_DEBUG}  → tool={tool_name} args={args}{COLOR_RESET}")

            # ── No tool → final answer (already streamed above) ──
            if not tool_name or str(tool_name).lower() == "null":
                logger.debug(f"Agent response: {message}")
                history.append({"role": "assistant", "content": message})
                mem.add_interaction(user_input, message)
                responded = True
                break

            # ── Duplicate tool call guard ──
            call_signature = (tool_name, str(args))
            if call_signature == last_tool_call:
                print(f"\n{COLOR_AI}Agent:{COLOR_RESET} Detected a repeated call to '{tool_name}' — stopping to avoid a loop.")
                logger.warning(f"Duplicate tool call: {call_signature}")
                responded = True
                break
            last_tool_call = call_signature

            # ── Tool not in registry ──
            if tool_name not in AVAILABLE_TOOLS:
                available = ", ".join(AVAILABLE_TOOLS.keys())
                print(
                    f"\n{COLOR_AI}Agent:{COLOR_RESET} Unknown tool '{tool_name}'. "
                    f"Available: {available}"
                )
                logger.warning(f"Unknown tool: '{tool_name}'")
                responded = True
                break

            # ── Execute tool ──
            if DEBUG_MODE:
                print(f"{COLOR_DEBUG}[*] Calling {tool_name}({', '.join(map(str, args))})...{COLOR_RESET}")
            logger.debug(f"Executing: {tool_name}({args})")

            try:
                tool_result = AVAILABLE_TOOLS[tool_name](*args)
                print(f"\n{COLOR_AI}[{tool_name}]{COLOR_RESET} {tool_result}")
                logger.debug(f"Tool result [{tool_name}]: {str(tool_result)[:300]}")
            except TypeError as e:
                tool_result = f"Tool '{tool_name}' was called with wrong arguments: {e}."
                logger.error(f"TypeError in '{tool_name}': {e}")
                print(f"\n{COLOR_AI}Agent:{COLOR_RESET} {tool_result}")
            except Exception as e:
                tool_result = f"Unexpected error in '{tool_name}': {e}."
                logger.exception(f"Error in '{tool_name}': {e}")
                print(f"\n{COLOR_AI}Agent:{COLOR_RESET} {tool_result}")

            mem.update(tool_name, str(tool_result), args)

            # Feed result back into messages so the model can chain tools
            tool_msg = f"[Tool: {tool_name}] Result:\n{tool_result}"
            messages.append({"role": "assistant", "content": tool_msg})
            messages.append({
                "role": "user",
                "content": (
                    "Tool result above is complete. "
                    "If the task is done, respond with tool null and summarise. "
                    "If another tool is needed, call it now."
                )
            })

        else:
            # Loop exhausted → max steps reached
            if not responded:
                print(
                    f"\n{COLOR_AI}Agent:{COLOR_RESET} Reached the {MAX_STEPS}-step limit "
                    "without a final answer. Please try a simpler request."
                )
                logger.warning(f"Max steps ({MAX_STEPS}) reached.")

        # Keep history tidy — record last tool exchange as a summary entry
        if not responded:
            history.append({"role": "assistant", "content": f"[Used tool chain for: {user_input}]"})


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    global DEBUG_MODE

    parser = argparse.ArgumentParser(description="Ollama Local AI Agent")
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug output (overrides DEBUG env var and log_level setting)"
    )
    parser.add_argument(
        "--config", default="settings.json",
        help="Path to settings JSON file (default: settings.json)"
    )
    args = parser.parse_args()

    DEBUG_MODE = args.debug or os.getenv("DEBUG", "false").strip().lower() == "true"

    config = load_config(args.config)
    logger = setup_logger(config["log_file"], config["log_level"], debug_mode=DEBUG_MODE)

    logger.info("=" * 50)
    logger.info(f"Agent session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config: {config}")

    run_agent(config, logger)

    logger.info("Agent session ended.")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()