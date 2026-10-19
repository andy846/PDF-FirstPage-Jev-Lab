"""Shared JSON POST transport with bounded retries and safe error reporting."""
from __future__ import annotations

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

RETRY_STATUS = {429, 500, 502, 503, 504}


def _error_tag(err: HTTPError) -> str:
    """Return only a short structured error type/code, never provider free text."""
    try:
        body = json.loads(err.read().decode("utf-8", "replace"))
    except (ValueError, OSError):
        return ""
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return ""
    tag = error.get("type") if isinstance(error.get("type"), str) else error.get("code")
    return f" ({tag})" if isinstance(tag, str) and 0 < len(tag) <= 60 else ""


def post_json(url: str, payload: dict, headers: dict, *, timeout: float, label: str,
              retries: int = 2, backoff: float = 0.8) -> dict:
    """POST JSON, retrying transient failures; raise RuntimeError with a safe summary."""
    request = Request(url, data=json.dumps(payload).encode("utf-8"), method="POST", headers=headers)
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                try:
                    return json.load(response)
                except ValueError:
                    raise RuntimeError(f"{label} returned invalid JSON; no decisions recorded.") from None
        except HTTPError as err:
            if err.code in RETRY_STATUS and attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise RuntimeError(f"{label} HTTP {err.code}{_error_tag(err)}; no decisions recorded.") from None
        except (URLError, TimeoutError, ConnectionError):
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise RuntimeError(f"Unable to reach {label}; no decisions recorded.") from None
    raise RuntimeError(f"Unable to reach {label}; no decisions recorded.")


def note_metadata(data, models: list[str], providers: list[str]) -> None:
    """Collect served model/provider metadata without duplicating entries."""
    if not isinstance(data, dict):
        return
    model = data.get("model")
    if isinstance(model, str) and model and model not in models:
        models.append(model)
    provider = data.get("provider")
    if isinstance(provider, str) and provider and provider not in providers:
        providers.append(provider)


def model_matches(requested: str, served: list[str]) -> bool:
    """True when a served model is the requested ID or a pinned version of it."""
    return any(item == requested or item.startswith(requested + "-") for item in served)
