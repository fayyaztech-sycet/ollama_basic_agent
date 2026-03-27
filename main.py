import sys
import os
import re
from ollama_service import OllamaService
from tools import AVAILABLE_TOOLS

SYSTEM_PROMPT = """You are a Strict Linux System Agent. You ONLY operate through your provided tools.

STRICT RULES:
1. If a user request is NOT covered by a tool, you MUST say: "I am sorry, but I can only perform actions via my authorized tools, and this request is not supported."
2. NEVER provide manual terminal commands or instructions for the user to run themselves.
3. NEVER give advice on how to perform tasks outside your toolset.
4. Format: <tool_call>tool_name("arg1", "arg2")</tool_call> (or use markdown blocks).
5. When a tool returns a result, provide the data to the user naturally.

AVAILABLE TOOLS:
- check_updates(): Checks for system updates.
- download_youtube(url): Downloads a YouTube video.
- convert_video(input_file, output_format): Converts video files.
- get_system_status(): Returns CPU usage, RAM usage, and top processes.
- gpu_status(): Returns detailed GPU usage (nvidia-smi).
- run_safe_command(base_cmd, *args): Runs a whitelisted command with any arguments (available bases: "ls", "cat", "df", "uptime", "nvidia-smi", "yt-dlp", "ffmpeg").

EXAMPLES:
User: show all files
Assistant: run_safe_command("ls", "-la")

User: check available formats for [URL]
Assistant: run_safe_command("yt-dlp", "-F", "[URL]")
"""

def main():
    service = OllamaService()
    
    print("Checking for available Ollama models...")
    models = service.list_models()
    
    if not models:
        print("No models found. Please make sure Ollama is running.")
        sys.exit(1)
        
    print("\nAvailable Models:")
    for i, model in enumerate(models):
        print(f"{i + 1}. {model}")
        
    choice = input("\nSelect a model (number) or press Enter for the first one: ").strip()
    selected_model = models[int(choice) - 1] if choice.isdigit() and 0 < int(choice) <= len(models) else models[0]
            
    print(f"\nUsing model: {selected_model}")
    print("Agent is ready! (type 'quit' to exit)")
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
            
        messages.append({"role": "user", "content": user_input})
        
        print("\nAgent: ", end="", flush=True)
        
        full_response = ""
        for chunk in service.chat(selected_model, messages, stream=True):
            print(chunk, end="", flush=True)
            full_response += chunk
        print()

        # Robust tool detection
        # 1. XML Tag: <tool_call>name()</tool_call>
        # 2. Markdown Block: ```bash name() ```
        # 3. Standalone: name() on a single line
        tool_match = re.search(r"<tool_call>(\w+)\((.*?)\)</tool_call>", full_response)
        if not tool_match:
            tool_match = re.search(r"```(?:bash|python)?\s*(\w+)\((.*?)\)\s*```", full_response)
        if not tool_match:
            # Matches name("arg") if it's the only thing on a line (whitespace allowed)
            tool_match = re.search(r"^\s*(\w+)\((.*?)\)\s*$", full_response, re.MULTILINE)
            
        if tool_match:
            tool_name = tool_match.group(1)
            args_str = tool_match.group(2)
            args = re.findall(r'"([^"]*)"', args_str)
            
            if tool_name in AVAILABLE_TOOLS:
                print(f"[*] Executing {tool_name}...")
                try:
                    tool_output = AVAILABLE_TOOLS[tool_name](*args)
                    print(f"[*] Result: {tool_output}")
                    
                    # Store history and provide neutral feedback
                    messages.append({"role": "assistant", "content": full_response})
                    messages.append({"role": "user", "content": f"[SYSTEM]: Tool {tool_name} returned: {tool_output}. Please provide the resulting information to the user naturally."})
                    
                    print("Agent: ", end="", flush=True)
                    final_response = ""
                    for chunk in service.chat(selected_model, messages, stream=True):
                        print(chunk, end="", flush=True)
                        final_response += chunk
                    print()
                    messages.append({"role": "assistant", "content": final_response})
                except TypeError as e:
                    print(f"[!] Error: {e}")
            else:
                print(f"[!] {tool_name} not found.")
        else:
            messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
