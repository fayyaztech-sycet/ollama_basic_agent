import sys
import os
import re
import json
from ollama_service import OllamaService
from tools import AVAILABLE_TOOLS

SYSTEM_PROMPT = """You are a Strict Linux System Agent. You MUST respond in valid JSON format ONLY.

RESPONSE FORMAT:
{
  "thought": "Your internal reasoning for this step.",
  "tool": "tool_name or null",
  "args": ["arg1", "arg2"]
}

STRICT RULES:
1. ONLY respond with the JSON object. No extra text, no markdown blocks unless it is ONLY the JSON.
2. If the user request is NOT covered by a tool, set "tool": null and provide the reason in "thought".
3. NEVER provide manual terminal commands or instructions for the user to run themselves.
4. "args" MUST be a list of strings, even if empty.
5. All tool calls MUST be whitelisted.

AVAILABLE TOOLS:
- check_updates(): Checks for system updates.
- download_youtube(url): Downloads a YouTube video.
- convert_video(input_file, output_format): Converts video files.
- get_system_status(): Returns CPU usage, RAM usage, and top processes.
- gpu_status(): Returns detailed GPU usage (nvidia-smi).
- run_safe_command(base_cmd, *args): Runs a whitelisted command (available: "ls", "cat", "df", "uptime", "nvidia-smi", "yt-dlp", "ffmpeg").
"""

def extract_json(text):
    """Extract and parse JSON from the LLM response."""
    try:
        # Try finding JSON between {} if it exists or parse whole
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except Exception:
        return None

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
        
        retries = 1
        while retries >= 0:
            print("\nAgent Thought: ", end="", flush=True)
            
            full_raw_response = ""
            for chunk in service.chat(selected_model, messages, stream=True):
                full_raw_response += chunk
            
            data = extract_json(full_raw_response)
            
            if data:
                print(data.get("thought", "..."))
                
                tool_name = data.get("tool")
                args = data.get("args", [])
                
                if tool_name:
                    if tool_name in AVAILABLE_TOOLS:
                        print(f"[*] Executing {tool_name} with {args}...")
                        try:
                            # Validation: args must be list
                            if not isinstance(args, list):
                                raise ValueError("Args must be a list.")
                                
                            tool_result = AVAILABLE_TOOLS[tool_name](*args)
                            print(f"[*] Result: {tool_result}")
                            
                            messages.append({"role": "assistant", "content": full_raw_response})
                            messages.append({"role": "user", "content": f"[SYSTEM]: Tool {tool_name} returned: {tool_result}. Respond with the final information in JSON format."})
                            # Clear retries and loop once more to get final natural response
                            retries = 0
                            continue
                        except Exception as e:
                            print(f"[!] Error: {e}")
                            break
                    else:
                        print(f"[!] Tool {tool_name} not found.")
                        break
                else:
                    # No tool, just a final thought or information
                    # We might want to print the thought to the user if it contains the actual answer
                    # Or adjust the prompt format. For now, let's treat "thought" as the payload.
                    if not tool_name and data.get("thought") and not any(m['role'] == 'assistant' for m in messages[-2:]):
                         messages.append({"role": "assistant", "content": full_raw_response})
                    break
            else:
                if retries > 0:
                    print("[!] Failed to parse JSON. Retrying...")
                    messages.append({"role": "user", "content": "Error: Your response was not a valid JSON object. Please respond ONLY in the required JSON format."})
                else:
                    print("[!] Persistent JSON parsing error.")
                retries -= 1

if __name__ == "__main__":
    main()
