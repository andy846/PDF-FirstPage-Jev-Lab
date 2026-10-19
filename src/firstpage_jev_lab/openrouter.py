"""OpenRouter-hosted model backend for per-page first-page pseudo-labels.

Text leaves the machine only after explicit opt-in. Replies are uncalibrated
model probabilities, never verified ground truth.
"""
from __future__ import annotations

import json
import math
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .jev import redact
from .pdftext import PageText

OFFICIAL_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "typesafe/jev-router"
USER_AGENT = "pdf-firstpage-jev-lab/0.2.0 (+https://github.com/andy846/PDF-FirstPage-Jev-Lab)"
SYSTEM_PROMPT = ("You classify pages from a multi-letter text PDF. For each page, estimate the probability "
                 "(0 to 1) that it is the FIRST page of a distinct new mailpiece rather than a continuation "
                 'page. Reply with strict JSON only: {"pages": {"<page number>": <probability>, ...}}')
_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def parse_pages(content) -> dict[int, float]:
    """Validate a chat reply and return source page -> first-page probability."""
    if not isinstance(content, str):
        raise ValueError("OpenRouter returned no text content.")
    try:
        data = json.loads(_CODE_FENCE.sub("", content.strip()))
    except json.JSONDecodeError:
        raise ValueError("OpenRouter reply was not valid JSON; no decisions recorded.") from None
    pages = data.get("pages") if isinstance(data, dict) else None
    if not isinstance(pages, dict) or not pages:
        raise ValueError("OpenRouter reply has no usable pages mapping; no decisions recorded.")
    output: dict[int, float] = {}
    for key, value in pages.items():
        try:
            number = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid page key {key!r} in the OpenRouter reply.") from None
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid first-page probability for source page {number}.")
        output[number] = float(value)
    return output


class OpenRouterClassifier:
    def __init__(self, *, key: str | None = None, model: str | None = None,
                 allow_external_text: bool = False, transport=None, endpoint: str | None = None):
        if not allow_external_text:
            raise ValueError("OpenRouter requires explicit --allow-external-text permission.")
        self.key = key or os.getenv("OPENROUTER_API_KEY")
        if not self.key:
            raise ValueError("Set OPENROUTER_API_KEY for OpenRouter (never put it in source code).")
        self.model = model or os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL
        self.endpoint = endpoint or os.getenv("OPENROUTER_ENDPOINT") or OFFICIAL_ENDPOINT
        self.transport = transport or self._http

    def _http(self, payload: dict) -> dict:
        request = Request(self.endpoint, data=json.dumps(payload).encode("utf-8"), method="POST",
                          headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                                   "User-Agent": USER_AGENT, "X-Title": "PDF FirstPage Jev Lab"})
        try:
            with urlopen(request, timeout=120) as response:
                return json.load(response)
        except HTTPError as err:
            # Do not reveal provider error response; it may echo customer content.
            raise RuntimeError(f"OpenRouter HTTP {err.code}; no decisions recorded.") from None
        except URLError:
            raise RuntimeError("Unable to reach OpenRouter; no decisions recorded.") from None

    def classify(self, pages: list[PageText], *, batch_size: int = 5,
                 chars_per_page: int = 1800) -> dict[int, float]:
        if batch_size < 1 or batch_size > 20:
            raise ValueError("batch_size must be 1..20")
        output: dict[int, float] = {}
        for start in range(0, len(pages), batch_size):
            batch = pages[start:start+batch_size]
            listing = "\n".join(f"Page {p.number}: {redact(p.text)[:chars_per_page]}" for p in batch)
            payload = {
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": listing}],
            }
            data = self.transport(payload)
            choices = data.get("choices") if isinstance(data, dict) else None
            content = (choices[0].get("message", {}).get("content")
                       if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None)
            probabilities = parse_pages(content)
            for p in batch:
                if p.number not in probabilities:
                    raise ValueError(f"Missing OpenRouter decision for source page {p.number}.")
                output[p.number] = probabilities[p.number]
        return output
