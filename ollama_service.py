import requests
import json

class OllamaService:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def list_models(self):
        """Fetch available models from Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []

    def generate_response(self, model, prompt, stream=False):
        """Get a response from a specific model."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        try:
            if stream:
                response = requests.post(url, json=payload, stream=True)
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            yield chunk['response']
                        if chunk.get('done'):
                            break
            else:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                return response.json().get('response', '')
        except Exception as e:
            if stream:
                yield f"Error generating response: {e}"
            else:
                return f"Error generating response: {e}"

    def chat(self, model, messages, stream=False):
        """Chat with a specific model using a message history."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        try:
            if stream:
                response = requests.post(url, json=payload, stream=True)
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if 'message' in chunk and 'content' in chunk['message']:
                            yield chunk['message']['content']
                        if chunk.get('done'):
                            break
            else:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                return response.json().get('message', {}).get('content', '')
        except Exception as e:
            if stream:
                yield f"Error in chat: {e}"
            else:
                return f"Error in chat: {e}"
