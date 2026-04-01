import os
import requests
import json
import psutil
import socket
import subprocess
import shlex
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # loads .env if present, silently skipped if not

OLLAMA_URL      = os.getenv("OLLAMA_URL",     "http://localhost:11434/api/chat")
OLLAMA_BASE_URL = OLLAMA_URL.rsplit("/api/", 1)[0]   # e.g. http://localhost:11434
DEFAULT_MODEL   = os.getenv("DEFAULT_MODEL",  "llama3")
MAX_HISTORY     = int(os.getenv("MAX_HISTORY", "20"))
_raw_cmds       = os.getenv("ALLOWED_SHELL_COMMANDS", "ls,pwd,echo,date,whoami,hostname,uname")

# ---------------- TOOLS ---------------- #

def get_time():
    return datetime.now().strftime("%A, %B %d %Y — %I:%M %p")

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def get_ram_usage():
    mem = psutil.virtual_memory()
    return f"{mem.percent}% used ({mem.used // (1024**2)} MB / {mem.total // (1024**2)} MB)"

def get_cpu_usage():
    cpu = psutil.cpu_percent(interval=1)
    cores = psutil.cpu_count(logical=True)
    return f"{cpu}% across {cores} logical cores"

def get_disk_usage():
    disk = psutil.disk_usage("/")
    return (
        f"{disk.percent}% used  "
        f"({disk.used // (1024**3)} GB used / {disk.total // (1024**3)} GB total)"
    )

# Restricted shell — allow-list sourced from .env / default
ALLOWED_COMMANDS = {c.strip() for c in _raw_cmds.split(",") if c.strip()}

def run_shell(command: str):
    try:
        parts = shlex.split(command)
    except ValueError as e:
        return f"[shell] Invalid command syntax: {e}"

    if not parts:
        return "[shell] Empty command."

    if parts[0] not in ALLOWED_COMMANDS:
        return (
            f"[shell] Command '{parts[0]}' is not allowed. "
            f"Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"
        )

    try:
        result = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=5
        )
        output = result.stdout.strip() or result.stderr.strip()
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "[shell] Command timed out."
    except Exception as e:
        return f"[shell] Error: {e}"

TOOLS = {
    "get_time":      get_time,
    "get_local_ip":  get_local_ip,
    "get_ram_usage": get_ram_usage,
    "get_cpu_usage": get_cpu_usage,
    "get_disk_usage": get_disk_usage,
}

# ---------------- SHORTCUTS ---------------- #

SHORTCUTS = {
    "/time":  get_time,
    "/ip":    get_local_ip,
    "/ram":   get_ram_usage,
    "/cpu":   get_cpu_usage,
    "/disk":  get_disk_usage,
    "/help":  lambda: (
        "Shortcuts: /time /ip /ram /cpu /disk /clear /help\n"
        "Shell:     /sh <command>  (allowed: " + ", ".join(sorted(ALLOWED_COMMANDS)) + ")\n"
        "Type 'exit' or 'quit' to leave."
    ),
    "/clear": lambda: "__CLEAR__",
}

def handle_shortcut(user_input: str):
    """Return (handled: bool, output: str)."""
    token = user_input.split()[0].lower()

    if token == "/sh":
        cmd = user_input[3:].strip()
        return True, run_shell(cmd)

    if token in SHORTCUTS:
        result = SHORTCUTS[token]()
        if result == "__CLEAR__":
            return True, "__CLEAR__"
        return True, result

    return False, ""

# ---------------- LLM ---------------- #

def chat(messages, model):
    res = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": messages,
        "stream": False
    })
    res.raise_for_status()
    return res.json()["message"]["content"].strip()

# ---------------- AGENT ---------------- #

SYSTEM_PROMPT = """\
You are a helpful CLI AI agent with access to system tools.

Available tools:
- get_time       → current date and time
- get_local_ip   → local network IP address
- get_ram_usage  → RAM usage info
- get_cpu_usage  → CPU usage info
- get_disk_usage → disk usage info

STRICT RULES:
1. If the user's request requires a tool, respond with ONLY valid JSON (no extra text):
   {"tool": "tool_name"}

2. If no tool is needed, respond normally in plain text.

Do NOT include markdown or explanation when returning JSON.\
"""

def run_agent(user_input: str, model: str, history: list):
    history.append({"role": "user", "content": user_input})

    # Trim history to MAX_HISTORY most-recent turns (pairs of user+assistant)
    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    reply = chat(messages, model)

    # Attempt tool dispatch
    try:
        # Strip markdown code fences if model wraps JSON
        clean = reply.strip().strip("```json").strip("```").strip()
        data = json.loads(clean)
        tool_name = data.get("tool")

        if tool_name in TOOLS:
            tool_result = TOOLS[tool_name]()

            # Ask the LLM to humanise the raw result
            final = chat([
                {
                    "role": "system",
                    "content": "Turn this raw system data into one friendly, natural sentence. No markdown."
                },
                {"role": "user", "content": tool_result}
            ], model)

            history.append({"role": "assistant", "content": final})
            return final

    except (json.JSONDecodeError, AttributeError, KeyError):
        pass

    history.append({"role": "assistant", "content": reply})
    return reply

# ---------------- CLI LOOP ---------------- #

def list_models():
    """Fetch installed models from Ollama. Returns list of name strings."""
    try:
        res = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        res.raise_for_status()
        return [m["name"] for m in res.json().get("models", [])]
    except Exception:
        return []


def pick_model():
    """Interactive model selector. Falls back to DEFAULT_MODEL on error."""
    models = list_models()

    if not models:
        print(f"[warning] Could not reach Ollama or no models installed.")
        print(f"          Falling back to default model: {DEFAULT_MODEL}\n")
        return DEFAULT_MODEL

    # Mark the default with an asterisk
    default_idx = next(
        (i for i, m in enumerate(models) if m.startswith(DEFAULT_MODEL)),
        0
    )

    print("Available models:")
    for i, name in enumerate(models):
        marker = " *" if i == default_idx else ""
        print(f"  [{i + 1}] {name}{marker}")
    print(f"  (* = default from .env)\n")

    while True:
        choice = input(f"Pick a model [1-{len(models)}] or press Enter for default: ").strip()
        if choice == "":
            return models[default_idx]
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print(f"  Please enter a number between 1 and {len(models)}.")


def main():
    print("\n╔══════════════════════════════╗")
    print("║      AI CLI AGENT  v2.0      ║")
    print("╚══════════════════════════════╝")
    print("Type /help for shortcuts or 'exit' to quit.\n")

    model = pick_model()
    print(f"\nUsing model: {model}\n{'─'*34}\n")

    history = []   # conversation memory

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # --- shortcuts (fast path, no LLM call) ---
        handled, shortcut_result = handle_shortcut(user_input)
        if handled:
            if shortcut_result == "__CLEAR__":
                history.clear()
                print("[memory cleared]\n")
            else:
                print(f"→ {shortcut_result}\n")
            continue

        # --- agent (LLM path) ---
        try:
            response = run_agent(user_input, model, history)
            print(f"AI: {response}\n")
        except requests.exceptions.ConnectionError:
            print("AI: [error] Cannot reach Ollama. Is it running? (ollama serve)\n")
        except Exception as e:
            print(f"AI: [error] {e}\n")


if __name__ == "__main__":
    main()
