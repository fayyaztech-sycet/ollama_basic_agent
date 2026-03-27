import sys
import os
import re
import json
import logging
from datetime import datetime
from ollama_service import OllamaService
from tools import AVAILABLE_TOOLS

# ─────────────────────────────────────────────
# CONFIG LOADER
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    "ollama_host": "http://localhost:11434",
    "max_steps": 5,
    "history_window": 14,
    "log_file": "agent.log",
    "log_level": "INFO",
    "memory_file": "memory.json"
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

BASE_SYSTEM_PROMPT = """You are a Strict Linux System Agent with persistent memory. You MUST respond in valid JSON format ONLY.

RESPONSE FORMAT:
{{
  "thought": "Your internal reasoning for this step.",
  "tool": "tool_name or null",
  "args": ["arg1", "arg2"]
}}

STRICT RULES:
1. ONLY respond with the JSON object. No extra text.
2. If a request is a general question (Knowledge/Why/What) not requiring a tool, set "tool": null and provide a detailed explanation in "thought".
3. NEVER provide manual terminal commands or instructions for the user to run.
4. "args" MUST always be a list, even if empty: [].
5. PATIENCE: Never open a file or run a command unless explicitly asked. If you find something, report it first.

AVAILABLE TOOLS:
- check_updates(): Checks for system updates.
- download_youtube(url): Downloads a YouTube video.
- convert_video(input_file, output_format): Converts video files.
- get_system_status(): Returns CPU/RAM usage.
- gpu_status(): Returns GPU usage.
- list_directory(path=".", show_sizes=False, include_dir_size=False): List files/folders.
- open_file(path): Opens a file or directory using the default system handler.
- run_safe_command(base_cmd, *args): Runs whitelisted command (ls, cat, df, uptime, nvidia-smi, yt-dlp, ffmpeg).

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
    """Extract and parse the first JSON object found in LLM output."""
    try:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


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
    print("\nAgent is ready! (type 'quit' or 'exit' to stop)\n")
    print("─" * 50)

    MAX_STEPS     = config["max_steps"]
    HISTORY_WINDOW = config["history_window"]
    history: list[dict] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[Agent] Goodbye!")
            logger.info("Session ended by user (KeyboardInterrupt / EOF).")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("[Agent] Goodbye!")
            logger.info("Session ended by user command.")
            break

        logger.info(f"User: {user_input}")
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
            print(f"\r[Step {step}/{MAX_STEPS}] Thinking...", end="", flush=True)
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
                        "content": "Error: Your last response was not valid JSON. Respond ONLY with the JSON object."
                    })

            if not data:
                logger.error(f"Step {step}: could not parse JSON after retry. Raw: {full_raw_response[:200]}")
                print(f"\n[!] The model returned an unreadable response. Please try rephrasing your request.")
                break

            thought    = data.get("thought", "")
            tool_name  = data.get("tool")
            args       = data.get("args", [])

            # Ensure args is always a list
            if not isinstance(args, list):
                args = [args]

            print(f"\r[Step {step}/{MAX_STEPS}] Thought: {thought}")
            logger.info(f"Step {step} | tool={tool_name} | args={args} | thought={thought[:80]}")

            # ── No tool → final answer ──
            if not tool_name or str(tool_name).lower() == "null":
                print(f"\nAgent: {thought}")
                logger.info(f"Agent response: {thought}")
                history.append({"role": "assistant", "content": full_raw_response})
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
                print(f"\n[!] {msg}")
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
                print(f"\n[!] Unknown tool '{tool_name}'.")
                logger.warning(f"Unknown tool requested: '{tool_name}'")
                print(f"Agent: {msg}")
                responded = True
                break

            # ── Execute tool ──
            print(f"[*] Calling {tool_name}({', '.join(map(str, args))})...")
            logger.info(f"Executing tool: {tool_name}({args})")

            try:
                tool_result = AVAILABLE_TOOLS[tool_name](*args)
                print(f"[*] Result: {tool_result}")
                logger.info(f"Tool result [{tool_name}]: {str(tool_result)[:300]}")
            except TypeError as e:
                tool_result = (
                    f"Tool '{tool_name}' was called with wrong arguments: {e}. "
                    "Please check argument types and try again."
                )
                logger.error(f"TypeError in tool '{tool_name}': {e}")
                print(f"\n[!] {tool_result}")
            except Exception as e:
                tool_result = (
                    f"An unexpected error occurred while running '{tool_name}': {e}."
                )
                logger.exception(f"Unexpected error in tool '{tool_name}': {e}")
                print(f"\n[!] {tool_result}")

            # Update memory and conversation
            mem.update(tool_name, str(tool_result), args)

            tool_msg = {"role": "user", "content": f"[SYSTEM]: Tool '{tool_name}' returned:\n{tool_result}"}
            history.append({"role": "assistant", "content": full_raw_response})
            history.append(tool_msg)
            messages.append({"role": "assistant", "content": full_raw_response})
            messages.append(tool_msg)

        else:
            # Loop exhausted without a break → max steps reached
            if not responded:
                msg = (
                    f"I reached the maximum reasoning limit ({MAX_STEPS} steps) "
                    "without a final answer. Please try a simpler or more specific request."
                )
                print(f"\nAgent: {msg}")
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