import base64
import json
import os
import secrets
import urllib.error
import urllib.request
from datetime import UTC, datetime


class GitHubPublisher:
    """Optional GitHub sync. It activates only when GITHUB_TOKEN is configured."""

    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.repository = os.getenv("GITHUB_REPOSITORY", "startupskala-source/MinhaIA_V4_Treinamento_Chat")
        self.branch = os.getenv("GITHUB_BRANCH", "master")

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def publish_fact(self, fact: str, session_id: str) -> str | None:
        if not self.enabled:
            return None
        date = datetime.now(UTC).date().isoformat()
        path = f"data/learned/{date}.jsonl"
        endpoint = f"https://api.github.com/repos/{self.repository}/contents/{path}"
        existing, sha = self._read_existing(endpoint)
        record = {
            "fact": fact,
            "session_id": session_id,
            "learned_at": datetime.now(UTC).isoformat(),
        }
        content = existing + json.dumps(record, ensure_ascii=False) + "\n"
        payload = {
            "message": "data: add learned chat memory",
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": self.branch,
        }
        if sha:
            payload["sha"] = sha
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="PUT",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": "MinhaIA-V4",
            },
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.loads(response.read())
        return result["content"]["html_url"]

    def _read_existing(self, endpoint: str) -> tuple[str, str | None]:
        request = urllib.request.Request(
            endpoint + f"?ref={self.branch}",
            headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {self.token}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return "", None
            raise
        return base64.b64decode(result["content"]).decode("utf-8"), result["sha"]


def teaching_key_matches(provided: str | None, expected: str | None) -> bool:
    return bool(provided and expected and secrets.compare_digest(provided, expected))
