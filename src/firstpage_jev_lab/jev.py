"""Official TypeSafe System One Jev API, plus an explicitly labeled offline mock."""
from __future__ import annotations

import math
import os
import re

from .pdftext import PageText
from .transport import note_metadata, post_json

OFFICIAL_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
# OpenCode Console serves the same System One model, including the free jev-1.13-free tier.
OPENCODE_ENDPOINT = "https://opencode.ai/inference/systemone/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"
OPENCODE_DEFAULT_MODEL = "jev-1.13"
# Cloudflare fronts the OpenCode gateway and bans the default Python-urllib signature (error 1010).
USER_AGENT = "pdf-firstpage-jev-lab/0.2.0 (+https://github.com/andy846/PDF-FirstPage-Jev-Lab)"
TASK_INSTRUCTIONS = ("Classify FIRST versus CONTINUATION pages in a multi-letter text PDF. "
                     "A first page begins a new independent letter; repeated headers can occur on continuation pages.")
PAGE_COUNTER = re.compile(r"\b(?:page\s*)?1\s*(?:of|/)\s*\d{1,4}\b", re.I)


def resolve_api_key() -> str | None:
    """Official TypeSafe key, or an OpenCode Console service key for the same model."""
    return os.getenv("TYPESAFE_API_KEY") or os.getenv("OPENCODE_API_KEY")


def page_question(number: int) -> dict:
    """System One noul question asking whether a source page starts a new mailpiece."""
    return {"type": "noul",
            "instructions": f"Is PDF source page {number} the FIRST page of a distinct new mailpiece, not a continuation page?"}


def redact(text: str) -> str:
    """Basic best-effort protection, NOT a verified data-loss-prevention solution."""
    text = re.sub(r"\b[\w.+'-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"\b(?:\d[ -]?){4,}\d\b", "[NUMBER]", text)
    text = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", "[DATE]", text)
    text = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", "[DATE]", text)
    return text


def _parse_answer(answer: dict, *, label: str = "Jev") -> float:
    if not isinstance(answer, dict) or answer.get("type", "noul") != "noul":
        raise ValueError(f"Unexpected {label} answer format (expected noul).")
    val = answer.get("noul")
    if type(val) not in (float, int) or not math.isfinite(val) or not 0 <= val <= 1:
        raise ValueError(f"Invalid {label} noul probability.")
    return float(val)


class JevClassifier:
    def __init__(self, *, key: str | None = None, model: str | None = None,
                 allow_external_text: bool = False, transport=None, endpoint: str | None = None):
        if not allow_external_text:
            raise ValueError("Jev requires explicit --allow-external-text permission.")
        self.key = key or resolve_api_key()
        if not self.key:
            raise ValueError("Set TYPESAFE_API_KEY (or OPENCODE_API_KEY) for the Jev API (never put it in source code).")
        self.endpoint = endpoint or os.getenv("JEV_ENDPOINT")
        if not self.endpoint:
            # An OpenCode-only key implies the OpenCode gateway and its model IDs.
            self.endpoint = (OPENCODE_ENDPOINT
                             if os.getenv("OPENCODE_API_KEY") and not os.getenv("TYPESAFE_API_KEY")
                             else OFFICIAL_ENDPOINT)
        default_model = OPENCODE_DEFAULT_MODEL if self.endpoint == OPENCODE_ENDPOINT else DEFAULT_MODEL
        self.model = model or os.getenv("JEV_MODEL") or default_model
        self.served_models: list[str] = []
        self.providers: list[str] = []
        self.transport = transport or self._http

    def _http(self, payload: dict) -> dict:
        return post_json(self.endpoint, payload,
                         {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                          "User-Agent": USER_AGENT},
                         timeout=35, label="TypeSafe Jev API")

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
                raise ValueError("Jev returned no answers. Results not recorded.")
            for p in batch:
                key = f"page_{p.number}"
                if key not in answers:
                    raise ValueError(f"Missing Jev decision for source page {p.number}.")
                output[p.number] = _parse_answer(answers[key])
        return output


class MockClassifier:
    """Offline fixture only: intentionally simple heuristics; never Jev measurements."""
    def classify(self, pages: list[PageText], **_unused) -> dict[int, float]:
        return {page.number: (0.98 if PAGE_COUNTER.search(page.text) else 0.02) for page in pages}
