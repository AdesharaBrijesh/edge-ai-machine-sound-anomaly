"""Shared helpers: paths, .env loading, Edge Impulse Studio API client and job polling."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STUDIO_API = "https://studio.edgeimpulse.com/v1/api"
INGESTION_API = "https://ingestion.edgeimpulse.com/api"


def load_env(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env without overriding variables already set."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_config() -> tuple[str, str]:
    """Return (api_key, project_id) or exit with a helpful message."""
    load_env()
    key = os.environ.get("EI_API_KEY", "").strip()
    pid = os.environ.get("EI_PROJECT_ID", "").strip()
    if not key or not pid:
        sys.exit("EI_API_KEY and EI_PROJECT_ID must be set in .env (see .env.example).")
    return key, pid


class EIClient:
    """Thin wrapper around the Edge Impulse Studio REST API (project-scoped)."""

    def __init__(self, api_key: str, project_id: str):
        self.project_id = project_id
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": api_key, "Accept": "application/json"})

    @classmethod
    def from_env(cls) -> "EIClient":
        return cls(*get_config())

    def url(self, path: str) -> str:
        path = path.lstrip("/")
        return f"{STUDIO_API}/{self.project_id}" + (f"/{path}" if path else "")

    def get(self, path: str = "", **params) -> dict:
        return self._check(self.session.get(self.url(path), params=params or None, timeout=120))

    def post(self, path: str, body: dict | None = None, **params) -> dict:
        return self._check(
            self.session.post(self.url(path), json=body if body is not None else {}, params=params or None, timeout=300)
        )

    def delete(self, path: str) -> dict:
        return self._check(self.session.delete(self.url(path), timeout=120))

    @staticmethod
    def _check(resp: requests.Response) -> dict:
        try:
            data = resp.json()
        except ValueError:
            raise RuntimeError(f"{resp.request.method} {resp.url} -> HTTP {resp.status_code}: {resp.text[:300]}")
        if resp.status_code >= 400 or (isinstance(data, dict) and data.get("success") is False):
            err = data.get("error", data) if isinstance(data, dict) else data
            raise RuntimeError(f"{resp.request.method} {resp.url} -> HTTP {resp.status_code}: {err}")
        return data

    def job_log_tail(self, job_id: int, lines: int = 25) -> str:
        try:
            out = self.get(f"jobs/{job_id}/stdout").get("stdout", [])
        except RuntimeError as exc:  # logs are best-effort
            return f"(could not fetch log: {exc})"
        out = sorted(out, key=lambda entry: str(entry.get("created", "")))
        text = "".join(entry.get("data", "") for entry in out)
        return "\n".join(text.rstrip().splitlines()[-lines:])

    def wait_job(self, job_id: int, label: str = "job", poll: float = 5.0, timeout: float = 3600.0) -> bool:
        """Poll a Studio job until it finishes. Returns True on success, raises on timeout."""
        start = time.time()
        last_report = 0.0
        while True:
            job = self.get(f"jobs/{job_id}/status").get("job", {})
            if job.get("finished"):
                ok = bool(job.get("finishedSuccessful"))
                took = time.time() - start
                print(f"[{label}] job {job_id} {'succeeded' if ok else 'FAILED'} after {took:.0f}s")
                if not ok:
                    print(self.job_log_tail(job_id))
                return ok
            if time.time() - start > timeout:
                raise TimeoutError(f"{label}: job {job_id} still running after {timeout:.0f}s")
            if time.time() - last_report > 60:
                print(f"[{label}] job {job_id} running... {time.time() - start:.0f}s")
                last_report = time.time()
            time.sleep(poll)
