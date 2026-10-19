"""Extract only embedded PDF text and spatial line information. No OCR."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import fitz

MM_PER_POINT = 25.4 / 72


@dataclass(frozen=True)
class TextLine:
    text: str
    region_mm: tuple[float, float, float, float]


@dataclass(frozen=True)
class PageText:
    number: int  # 1-based, source PDF order
    text: str
    lines: tuple[TextLine, ...]
    width_mm: float
    height_mm: float


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_pages(path: str | Path) -> list[PageText]:
    path = Path(path)
    result: list[PageText] = []
    with fitz.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError("Encrypted PDFs are unsupported by this prototype.")
        if not pdf.page_count:
            raise ValueError("The PDF has no pages.")
        for index, page in enumerate(pdf):
            lines: list[TextLine] = []
            raw = page.get_text("dict")
            for block in raw.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    text = normalize(" ".join(span.get("text", "") for span in line.get("spans", [])))
                    if not text:
                        continue
                    x0, y0, x1, y1 = line["bbox"]
                    lines.append(TextLine(text, (x0 * MM_PER_POINT, y0 * MM_PER_POINT,
                                                  (x1-x0) * MM_PER_POINT, (y1-y0) * MM_PER_POINT)))
            text = "\n".join(item.text for item in lines)
            result.append(PageText(index + 1, text, tuple(lines),
                                   page.rect.width * MM_PER_POINT, page.rect.height * MM_PER_POINT))
    if not any(p.text for p in result):
        raise ValueError("No selectable PDF text was found. Scans/images are out of scope.")
    return result


def select_sample(pages: list[PageText], count: int, method: str) -> list[PageText]:
    if count < 2:
        raise ValueError("Select at least two sample pages.")
    if method == "first":
        return pages[:count]
    if method != "distributed":
        raise ValueError("Unknown sample method")
    if count >= len(pages):
        return pages[:]
    indices = [round(i * (len(pages)-1) / (count-1)) for i in range(count)]
    return [pages[i] for i in dict.fromkeys(indices)]
