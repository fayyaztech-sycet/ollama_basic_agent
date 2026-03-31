import os
import json
import logging
import requests

logger = logging.getLogger("agent")


class OllamaService:
    """
    Client for the Ollama local inference API.

    Configuration (in priority order):
      1. Argument passed to __init__
      2. OLLAMA_HOST environment variable
      3. Default: http://localhost:11434
    """

    # Timeouts: (connect_seconds, read_seconds)
    CONNECT_TIMEOUT  = 5
    DEFAULT_TIMEOUT  = (5, 120)   # short connect, long read for slow models
    STREAM_TIMEOUT   = (5, 300)   # extra time for streamed generation

    def __init__(self, base_url: str = None):
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")
        logger.debug(f"OllamaService initialised with base_url='{self.base_url}'")

    # ─────────────────────────────────────────
    # HEALTH
    # ─────────────────────────────────────────

    def is_available(self) -> bool:
        """
        Return True if Ollama is reachable, False otherwise.
        Use this at startup before any model calls.
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.CONNECT_TIMEOUT
            )
            response.raise_for_status()
            return True
        except requests.ConnectionError:
            logger.error(
                f"Cannot connect to Ollama at '{self.base_url}'. "
                "Is `ollama serve` running?"
            )
        except requests.Timeout:
            logger.error(
                f"Connection to Ollama at '{self.base_url}' timed out "
                f"after {self.CONNECT_TIMEOUT}s."
            )
        except requests.HTTPError as e:
            logger.error(f"Ollama health check returned HTTP error: {e}")
        return False

    # ─────────────────────────────────────────
    # MODEL LIST
    # ─────────────────────────────────────────

    def list_models(self) -> list[str]:
        """
        Return a list of locally available model names.
        Returns an empty list on any error.
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            logger.debug(f"Available models: {models}")
            return models
        except requests.Timeout:
            logger.error("Timed out while fetching model list from Ollama.")
        except requests.ConnectionError:
            logger.error("Connection error while fetching model list.")
        except requests.HTTPError as e:
            logger.error(f"HTTP error while fetching model list: {e}")
        except (KeyError, ValueError) as e:
            logger.error(f"Unexpected response format from /api/tags: {e}")
        return []

    # ─────────────────────────────────────────
    # CHAT  (used by the agent loop)
    # ─────────────────────────────────────────

    def chat(self, model: str, messages: list[dict], stream: bool = False):
        """
        Send a conversation to the /api/chat endpoint.

        Streaming mode  → generator that yields text chunks (str).
                          Raises RuntimeError on network/parse errors.
        Non-stream mode → returns the full reply as a str.
                          Returns empty string on error.
        """
        url     = f"{self.base_url}/api/chat"
        payload = {"model": model, "messages": messages, "stream": stream}

        if stream:
            return self._chat_stream(url, payload)
        else:
            return self._chat_blocking(url, payload)

    def _chat_stream(self, url: str, payload: dict):
        """Private generator for streaming chat responses."""
        try:
            response = requests.post(
                url,
                json=payload,
                stream=True,
                timeout=self.STREAM_TIMEOUT
            )
            response.raise_for_status()
        except requests.Timeout:
            raise RuntimeError(
                "The model took too long to start responding. "
                "Try a lighter model or increase STREAM_TIMEOUT."
            )
        except requests.ConnectionError as e:
            raise RuntimeError(f"Lost connection to Ollama during streaming: {e}")
        except requests.HTTPError as e:
            raise RuntimeError(f"Ollama returned HTTP error during chat: {e}")

        try:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping non-JSON stream line: {line!r} ({e})")
                    continue

                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content

                if chunk.get("done"):
                    logger.debug(
                        f"Stream done | "
                        f"eval_count={chunk.get('eval_count')} | "
                        f"eval_duration={chunk.get('eval_duration')}"
                    )
                    break

        except requests.exceptions.ChunkedEncodingError as e:
            raise RuntimeError(f"Stream was interrupted unexpectedly: {e}")
        except KeyboardInterrupt:
            raise

    def _chat_blocking(self, url: str, payload: dict) -> str:
        """Private method for non-streaming chat responses."""
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.Timeout:
            logger.error("Chat request timed out (non-streaming).")
        except requests.ConnectionError as e:
            logger.error(f"Connection error during chat: {e}")
        except requests.HTTPError as e:
            logger.error(f"HTTP error during chat: {e}")
        except (KeyError, ValueError) as e:
            logger.error(f"Unexpected response format from /api/chat: {e}")
        return ""

    # ─────────────────────────────────────────
    # GENERATE  (single-turn, no history)
    # ─────────────────────────────────────────

    def generate(self, model: str, prompt: str, stream: bool = False):
        """
        Send a single prompt to /api/generate (no conversation history).
        Useful for one-shot completions outside the agent loop.

        Streaming mode  → generator yielding text chunks.
        Non-stream mode → returns full reply as str.
        """
        url     = f"{self.base_url}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": stream}

        if stream:
            return self._generate_stream(url, payload)
        else:
            return self._generate_blocking(url, payload)

    def _generate_stream(self, url: str, payload: dict):
        """Private generator for streaming generate responses."""
        try:
            response = requests.post(
                url,
                json=payload,
                stream=True,
                timeout=self.STREAM_TIMEOUT
            )
            response.raise_for_status()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            raise RuntimeError(f"Error starting generate stream: {e}")

        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "response" in chunk:
                yield chunk["response"]
            if chunk.get("done"):
                break

    def _generate_blocking(self, url: str, payload: dict) -> str:
        """Private method for non-streaming generate responses."""
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except (requests.Timeout, requests.ConnectionError,
                requests.HTTPError, ValueError) as e:
            logger.error(f"Error in generate (blocking): {e}")
        return ""