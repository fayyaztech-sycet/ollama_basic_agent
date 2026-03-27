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
        
        last_tool_call = None
        
        # Multi-step reasoning loop (Max 5 steps)
        for step in range(1, 6):
            print(f"\n[Step {step}] Thinking...", end="", flush=True)
            
            # Retry logic for JSON parsing
            retries = 1
            data = None
            full_raw_response = ""
            
            while retries >= 0:
                full_raw_response = ""
                for chunk in service.chat(selected_model, messages, stream=True):
                    full_raw_response += chunk
                
                data = extract_json(full_raw_response)
                if data:
                    break
                else:
                    if retries > 0:
                        print("\n[!] JSON parse failed. Retrying...")
                        messages.append({"role": "user", "content": "Error: Invalid JSON. Respond ONLY in the required JSON format: {'thought': '...', 'tool': '...', 'args': [...]}"})
                    retries -= 1
            
            if not data:
                print("\n[!] Persistent JSON error. Stopping reasoning.")
                break
                
            # Clear "Thinking..." and print actual thought
            print(f"\r[Step {step}] Thought: {data.get('thought', '...')}")
            
            tool_name = data.get("tool")
            args = data.get("args", [])
            
            # Stop condition 1: No tool to call
            if not tool_name or tool_name.lower() == "null":
                break
                
            # Stop condition 2: Infinite loop detection
            if (tool_name, str(args)) == last_tool_call:
                print(f"[!] Warning: Repeated tool call ({tool_name}). Stopping to avoid infinite loop.")
                break
            
            last_tool_call = (tool_name, str(args))
            
            if tool_name in AVAILABLE_TOOLS:
                print(f"[*] Executing {tool_name} with {args}...")
                try:
                    if not isinstance(args, list):
                        raise ValueError("Args must be a list.")
                        
                    tool_result = AVAILABLE_TOOLS[tool_name](*args)
                    print(f"[*] Result: {tool_result}")
                    
                    # Update context for next reasoning step
                    messages.append({"role": "assistant", "content": full_raw_response})
                    messages.append({"role": "user", "content": f"[SYSTEM]: Tool {tool_name} returned: {tool_result}"})
                except Exception as e:
                    print(f"[!] Tool Error: {e}")
                    messages.append({"role": "user", "content": f"[SYSTEM]: Error executing {tool_name}: {e}"})
            else:
                print(f"[!] Tool {tool_name} not found.")
                messages.append({"role": "user", "content": f"[SYSTEM]: Tool {tool_name} is not in AVAILABLE_TOOLS."})

        if step == 5 and tool_name and tool_name.lower() != "null":
            print("\n[*] Reached max reasoning steps (5). Provide final summary.")
            # Final request for a summary if max steps reached
            messages.append({"role": "user", "content": "Max steps reached. Please provide a final summary of results now."})
            summary_response = ""
            for chunk in service.chat(selected_model, messages, stream=True):
                summary_response += chunk
            data_final = extract_json(summary_response)
            if data_final:
                print(f"Final Answer: {data_final.get('thought', '...')}")
            else:
                print(f"Final Output: {summary_response}")

if __name__ == "__main__":
    main()
