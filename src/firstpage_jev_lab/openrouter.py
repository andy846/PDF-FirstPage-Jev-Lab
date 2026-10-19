"""OpenRouter-hosted TypeSafe Jev decisions backend for per-page first-page pseudo-labels.

Uses OpenRouter's alpha decisions endpoint, which accepts the same state and
typed-question payload as the official TypeSafe System One API. Text leaves the
machine only after explicit opt-in. Replies are uncalibrated model
probabilities, never verified ground truth.
"""
from __future__ import annotations

import os

from .jev import TASK_INSTRUCTIONS, _parse_answer, page_question, redact
from .pdftext import PageText
from .transport import note_metadata, post_json

OFFICIAL_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.14"
USER_AGENT = "pdf-firstpage-jev-lab/0.2.0 (+https://github.com/andy846/PDF-FirstPage-Jev-Lab)"


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
        self.served_models: list[str] = []
        self.providers: list[str] = []
        self.transport = transport or self._http

    def _http(self, payload: dict) -> dict:
        return post_json(self.endpoint, payload,
                         {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                          "User-Agent": USER_AGENT, "X-Title": "PDF FirstPage Jev Lab"},
                         timeout=60, label="OpenRouter")

    def classify(self, pages: list[PageText], *, batch_size: int = 5,
                 chars_per_page: int = 1800) -> dict[int, float]:
        if batch_size < 1 or batch_size > 20:
            raise ValueError("batch_size must be 1..20")
        output: dict[int, float] = {}
        for start in range(0, len(pages), batch_size):
            batch = pages[start:start+batch_size]
            payload = {
                "model": self.model,
                "state": {"task": TASK_INSTRUCTIONS,
                          "pages": [{"page": p.number, "text": redact(p.text)[:chars_per_page]} for p in batch]},
                "questions": {f"page_{p.number}": page_question(p.number) for p in batch},
            }
            data = self.transport(payload)
            note_metadata(data, self.served_models, self.providers)
            answers = data.get("answers") if isinstance(data, dict) else None
            if not isinstance(answers, dict):
                raise ValueError("OpenRouter returned no answers. Results not recorded.")
            for p in batch:
                key = f"page_{p.number}"
                if key not in answers:
                    raise ValueError(f"Missing OpenRouter decision for source page {p.number}.")
                output[p.number] = _parse_answer(answers[key], label="OpenRouter")
        return output
