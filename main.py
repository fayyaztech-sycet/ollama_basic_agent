import sys
import os
import re
import json
import logging
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

DEBUG_MODE = os.getenv("DEBUG", "false").strip().lower() == "true"

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
                # Merge: user values override defaults
                return {**DEFAULT_CONFIG, **user_config}
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


# ─────────────────────────────────────────────
# LOGGER SETUP
# ─────────────────────────────────────────────

def setup_logger(log_file: str, log_level: str) -> logging.Logger:
    """Configure a logger that writes to both file and stdout."""
    logger = logging.getLogger("agent")
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid adding duplicate handlers on re-import
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except IOError as e:
        print(f"[Warning] Cannot write to log file '{log_file}': {e}")

    # Console handler (INFO and above only — keeps terminal clean)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))  # plain for console
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
1. Respond with the JSON object ONLY. Double-quoted keys and values. No markdown, no code fences, no extra text.
2. For greetings or general questions: set "tool": null, write the actual user-facing reply in "message".
3. For actionable requests (status, ping, list files, etc.): set "tool" to the correct tool name and call it — do NOT just describe what you would do.
4. NEVER provide manual terminal commands for the user to run yourself.
5. "args" MUST always be a JSON array, even when empty: [].
6. FLAGS vs PATHS: When passing flags like -h, -l, -a to run_safe_command, keep them as-is (e.g. ["df", "-h"] not ["-h"] as a path). Flags always come before paths.
7. CONTEXT: If the answer can be derived from a previous tool result already in the conversation, answer directly with tool null — do NOT call the tool again.
8. After calling a tool the result is shown immediately. Your job for that turn is done — do not add commentary unless the user asks.

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
- User: "hi" / "hello"           → {{"message": "Hello! I'm your local Linux AI assistant powered by Ollama. Here's what I can help you with:\n\n• System monitoring — CPU, RAM, GPU, disk, processes\n• File operations — search, list, read files\n• Network tools — ping, traceroute, interface status\n• Media — download YouTube, convert/resize/analyze images & videos\n• Process management — list, kill, restart processes\n• File transfer — download from URLs, upload via rsync\n• AI utilities — summarize or translate text\n• Scheduler — schedule tasks and set reminders\n\nWhat would you like to do?", "tool": null, "args": []}}
- User: "what's my CPU/RAM?"     → {{"message": "Fetching system status.", "tool": "get_system_status", "args": []}}
- User: "ping google.com"        → {{"message": "Pinging google.com.", "tool": "ping_host", "args": ["google.com"]}}
- User: "read ~/.bashrc"         → {{"message": "Reading .bashrc.", "tool": "run_safe_command", "args": ["cat", "~/.bashrc"]}}
- User: "what time is it?"       → {{"message": "Checking the time.", "tool": "run_safe_command", "args": ["date"]}}
- User: "list home directory"    → {{"message": "Listing home directory.", "tool": "run_safe_command", "args": ["ls", "~"]}}
- User: "check disk space"       → {{"message": "Checking disk space.", "tool": "run_safe_command", "args": ["df", "-h"]}}
- User: "show my username"       → {{"message": "Getting username.", "tool": "run_safe_command", "args": ["whoami"]}}
- User: "find *.mp4 files"       → {{"message": "Searching for mp4 files.", "tool": "search_files", "args": ["*.mp4"]}}
- User: "list processes"         → {{"message": "Listing running processes.", "tool": "list_processes", "args": []}}
- User: "network status"         → {{"message": "Checking network interfaces.", "tool": "network_status", "args": []}}

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
        return {"last_file": "None", "last_command": "None", "notes": []}

    def save(self):
        try:
            with open(self.filename, "w") as f:
                json.dump(self.memory, f, indent=2)
        except IOError as e:
            self.logger.warning(f"Could not save memory: {e}")

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
        return json.dumps(self.memory, indent=2)


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

    # 2. Greedy scan: find the largest {...} block spanning the whole object.
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

    MAX_STEPS     = config["max_steps"]
    HISTORY_WINDOW = config["history_window"]
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

        # Build message list for this turn (system + sliding window)
        messages = (
            [{"role": "system", "content": current_system_prompt}]
            + history[-(HISTORY_WINDOW):]
        )

        last_tool_call = None
        responded      = False

        for step in range(1, MAX_STEPS + 1):
            if DEBUG_MODE:
                print(f"{COLOR_DEBUG}\r[Step {step}/{MAX_STEPS}] Thinking...{COLOR_RESET}", end="", flush=True)
            logger.debug(f"Step {step}: sending {len(messages)} messages to model.")

            # ── LLM call with one JSON-retry ──
            full_raw_response = ""
            data = None

            for attempt in range(2):            # attempt 0 = normal, attempt 1 = retry
                full_raw_response = ""
                try:
                    for chunk in service.chat(selected_model, messages, stream=True):
                        full_raw_response += chunk
                except RuntimeError as e:
                    logger.error(f"Stream error on step {step}, attempt {attempt}: {e}")
                    break

                data = extract_json(full_raw_response)
                if data:
                    break

                if attempt == 0:
                    logger.warning(f"Step {step}: invalid JSON from model — retrying once.")
                    messages.append({
                        "role": "user",
                        "content": (
                            'Error: Your last response was not valid JSON. '
                            'Reply with ONLY a raw JSON object — no markdown, no code fences, no extra text. '
                            'Example: {"message": "your reply here", "tool": null, "args": []}'
                        )
                    })

            if not data:
                logger.error(f"Step {step}: could not parse JSON after retry. Raw: {full_raw_response[:200]}")
                print(f"\n{COLOR_AI}Agent:{COLOR_RESET} The model returned an unreadable response. Please try rephrasing your request.")
                break

            # Support both new 'message' field and legacy 'thought' field
            message    = data.get("message") or data.get("thought", "")
            tool_name  = data.get("tool")
            args       = data.get("args", [])

            # Ensure args is always a list
            if not isinstance(args, list):
                args = [args]

            logger.debug(f"Step {step} | tool={tool_name} | args={args} | message={message[:80]}")
            if DEBUG_MODE:
                print(f"{COLOR_DEBUG}[Step {step}] tool={tool_name} | args={args}{COLOR_RESET}")

            # ── No tool → final answer ──
            if not tool_name or str(tool_name).lower() == "null":
                print(f"\n{COLOR_AI}Agent:{COLOR_RESET} {message}")
                logger.debug(f"Agent response: {message}")
                history.append({"role": "assistant", "content": message})
                responded = True
                break

            # ── Duplicate tool call guard ──
            call_signature = (tool_name, str(args))
            if call_signature == last_tool_call:
                msg = (
                    f"I detected a repeated identical call to '{tool_name}' — "
                    "stopping to avoid an infinite loop. "
                    "Please rephrase your request or check the last tool result."
                )
                print(f"\n{COLOR_AI}Agent:{COLOR_RESET} {msg}")
                logger.warning(f"Duplicate tool call detected: {call_signature}")
                responded = True
                break
            last_tool_call = call_signature

            # ── Tool not in registry ──
            if tool_name not in AVAILABLE_TOOLS:
                available = ", ".join(AVAILABLE_TOOLS.keys())
                msg = (
                    f"I tried to use a tool called '{tool_name}', but it doesn't exist. "
                    f"My available tools are: {available}. "
                    "Please rephrase your request."
                )
                logger.warning(f"Unknown tool requested: '{tool_name}'")
                print(f"\n{COLOR_AI}Agent:{COLOR_RESET} {msg}")
                responded = True
                break

            # ── Execute tool ──
            logger.debug(f"Executing tool: {tool_name}({args})")
            if DEBUG_MODE:
                print(f"{COLOR_DEBUG}[*] Calling {tool_name}({', '.join(map(str, args))})...{COLOR_RESET}")

            try:
                tool_result = AVAILABLE_TOOLS[tool_name](*args)
                print(f"\n{COLOR_AI}Agent:{COLOR_RESET} {tool_result}")
                logger.debug(f"Tool result [{tool_name}]: {str(tool_result)[:300]}")
            except TypeError as e:
                tool_result = (
                    f"Tool '{tool_name}' was called with wrong arguments: {e}. "
                    "Please check argument types and try again."
                )
                logger.error(f"TypeError in tool '{tool_name}': {e}")
                print(f"\n{COLOR_AI}Agent:{COLOR_RESET} {tool_result}")
            except Exception as e:
                tool_result = (
                    f"An unexpected error occurred while running '{tool_name}': {e}."
                )
                logger.exception(f"Unexpected error in tool '{tool_name}': {e}")
                print(f"\n{COLOR_AI}Agent:{COLOR_RESET} {tool_result}")

            # Update memory and conversation
            mem.update(tool_name, str(tool_result), args)

            # Tool result is already printed — add to history and stop this turn
            history.append({"role": "assistant", "content": str(tool_result)})
            responded = True
            break

        else:
            # Loop exhausted without a break → max steps reached
            if not responded:
                msg = (
                    f"I reached the maximum reasoning limit ({MAX_STEPS} steps) "
                    "without a final answer. Please try a simpler or more specific request."
                )
                print(f"\n{COLOR_AI}Agent:{COLOR_RESET} {msg}")
                logger.warning(f"Max steps ({MAX_STEPS}) reached without response.")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    config = load_config("settings.json")
    logger = setup_logger(config["log_file"], config["log_level"])

    logger.info("=" * 50)
    logger.info(f"Agent session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config: {config}")

    run_agent(config, logger)

    logger.info("Agent session ended.")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()