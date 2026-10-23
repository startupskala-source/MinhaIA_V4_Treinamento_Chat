import json
import os
import urllib.error
import urllib.request


class NvidiaError(RuntimeError):
    pass


class NvidiaChat:
    """OpenAI-compatible NVIDIA NIM client using only the standard library."""

    endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(self):
        self.api_key = os.getenv("NVIDIA_API_KEY")
        self.model = os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def messages(
        self, history: list[tuple[str, str]], question: str, context: str = ""
    ) -> list[dict[str, str]]:
        system = (
            "Voce e MinhaIA, uma assistente em portugues. Responda de forma clara, honesta e util. "
            "Use o CONTEXTO LOCAL quando ele estiver presente. Nao invente fatos que nao estejam no contexto."
        )
        if context:
            system += f"\n\nCONTEXTO LOCAL:\n{context}"
        messages = [{"role": "system", "content": system}]
        for user_message, assistant_message in history[-6:]:
            messages.extend(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ]
            )
        messages.append({"role": "user", "content": question})
        return messages

    def complete(self, history: list[tuple[str, str]], question: str, context: str = "") -> str:
        if not self.enabled:
            raise NvidiaError("NVIDIA_API_KEY nao configurada.")
        payload = {
            "model": self.model,
            "messages": self.messages(history, question, context),
            "temperature": 0.35,
            "top_p": 0.9,
            "max_tokens": 600,
            "stream": False,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise NvidiaError(f"NVIDIA NIM respondeu HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise NvidiaError(f"Nao foi possivel conectar ao NVIDIA NIM: {error.reason}") from error
        try:
            return result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as error:
            raise NvidiaError("Resposta NVIDIA NIM em formato inesperado.") from error
