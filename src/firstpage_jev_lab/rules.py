"""Interpretable pattern and spatial rule mining; never executes AI-generated code."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re

from .pdftext import PageText, TextLine, normalize

PAGE_PATTERN = re.compile(r"\b(?:page\s*)?(\d{1,4})\s*(?:of|/)\s*(\d{1,4})\b", re.I)
MARKER_LIMIT = 100


@dataclass(frozen=True)
class Rule:
    kind: str  # "first_text" or "page_number"
    term: str | None = None
    region_mm: tuple[float, float, float, float] | None = None

    def to_detection_config(self) -> dict:
        rule = ({"kind": "page_number", "auto": True} if self.kind == "page_number"
                else {"kind": "first_text", "terms": [self.term]})
        if self.region_mm is not None:
            rule["region_mm"] = [round(v, 2) for v in self.region_mm]
        return {"version": 2, "combine": "any", "rules": [rule],
                "region_mm": None, "remove_separators": True, "profile_name": "CANDIDATE_ONLY"}

    def as_dict(self) -> dict:
        return {"kind": self.kind, "term": self.term, "region_mm": self.region_mm}


@dataclass
class Candidate:
    rule: Rule
    tp: int
    fp: int
    fn: int
    tn: int
    assessed_pages: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def score(self) -> tuple:
        # Avoid rules that split continuation pages; prefer spatial rules on ties.
        return (self.fp == 0, self.tp > 0, self.precision, self.recall,
                self.tp, -self.fp, self.rule.region_mm is not None)

    def to_dict(self) -> dict:
        return {**self.rule.as_dict(), "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                "precision_on_sample": round(self.precision, 4),
                "recall_on_sample": round(self.recall, 4),
                "assessed_pages": self.assessed_pages,
                "detection_config": self.rule.to_detection_config()}


def _static_term(text: str) -> str | None:
    """Conservatively mine static anchors; reject likely variable customer values."""
    line = normalize(text)
    if not line or len(line) > MARKER_LIMIT or '@' in line:
        return None
    match = re.match(r"^([\w\s/()&-]{4,55})\s*[:：]", line)
    if match:
        line = normalize(match.group(1))
    elif re.search(r"\d", line):
        return None
    if not (4 <= len(line) <= 55) or re.search(r"\d|\b(?:mr|mrs|ms)\.?\b", line, re.I):
        return None
    if len(line.split()) == 1 and len(line) < 7:
        return None
    if line.count(" ") > 7:
        return None
    if re.search(r"\b(?:name|address|tel|phone|email)\b", line, re.I):
        return None
    return line


def _within(line: TextLine, region: tuple[float, float, float, float]) -> bool:
    x, y, w, h = line.region_mm
    rx, ry, rw, rh = region
    cx, cy = x+w/2, y+h/2
    return rx <= cx <= rx+rw and ry <= cy <= ry+rh


def matches(page: PageText, rule: Rule) -> bool:
    lines = [line for line in page.lines if rule.region_mm is None or _within(line, rule.region_mm)]
    if rule.kind == "page_number":
        matches_ = [tuple(map(int, m.groups())) for line in lines for m in PAGE_PATTERN.finditer(line.text)]
        return len(matches_) == 1 and matches_[0][0] == 1 and matches_[0][1] >= 1
    if rule.kind == "first_text" and rule.term:
        return any(rule.term.casefold() in line.text.casefold() for line in lines)
    raise ValueError("Unsupported rule type")


def _region(boxes: list[tuple[float, float, float, float]], page: PageText):
    x0 = min(b[0] for b in boxes)-2
    y0 = min(b[1] for b in boxes)-2
    x1 = max(b[0]+b[2] for b in boxes)+2
    y1 = max(b[1]+b[3] for b in boxes)+2
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(page.width_mm, x1), min(page.height_mm, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1-x0, y1-y0)


def mine(sample_pages: list[PageText], probabilities: dict[int, float],
         *, positive_min: float = 0.8, negative_max: float = 0.2) -> list[Candidate]:
    """Jev predictions are pseudo-labels, never ground truth."""
    if negative_max >= positive_min:
        raise ValueError("positive_min must exceed negative_max")
    labels = {p.number: (True if probabilities[p.number] >= positive_min else
                         False if probabilities[p.number] <= negative_max else None) for p in sample_pages}
    positives = [p for p in sample_pages if labels[p.number] is True]
    negatives = [p for p in sample_pages if labels[p.number] is False]
    if len(positives) < 2 or len(negatives) < 1:
        raise ValueError("Not enough unambiguous pseudo-labels (need >=2 first pages and >=1 continuation).")
    terms = Counter()
    locations: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for page in positives:
        terms_seen = set()
        for line in page.lines:
            term = _static_term(line.text)
            if term:
                if term not in terms_seen:
                    terms[term] += 1
                    terms_seen.add(term)
                locations[term].append(line.region_mm)
    rules: list[Rule] = [Rule("page_number")]
    for term, support in terms.most_common(90):
        if support < 2:
            continue
        rules.append(Rule("first_text", term))
        box = _region(locations[term], positives[0])
        if box:
            rules.append(Rule("first_text", term, box))
    candidates: list[Candidate] = []
    for rule in rules:
        tp = sum(matches(p, rule) for p in positives)
        fp = sum(matches(p, rule) for p in negatives)
        if not tp:
            continue
        candidates.append(Candidate(rule, tp, fp, len(positives)-tp,
                                    len(negatives)-fp, len(positives)+len(negatives)))
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def make_index(pages: list[PageText], rule: Rule) -> tuple[list[tuple[int, int]], list[int]]:
    starts = [p.number for p in pages if matches(p, rule)]
    if not starts or starts[0] != 1:
        raise ValueError("First source page has no rule match: cannot infer a safe index.")
    if sorted(set(starts)) != starts:
        raise ValueError("Duplicate or disordered source page boundaries")
    groups = [(start, (starts[i+1]-1 if i+1 < len(starts) else len(pages)))
              for i, start in enumerate(starts)]
    assert sum(end-start+1 for start, end in groups) == len(pages)
    return groups, starts


def evaluate(starts: list[int], ground_truth: set[int]) -> dict:
    predicted = set(starts)
    tp = len(predicted & ground_truth)
    fp = len(predicted-ground_truth)
    fn = len(ground_truth-predicted)
    return {"true_starts": tp, "false_splits": fp, "missed_starts": fn,
            "precision": round(tp/len(predicted), 4) if predicted else 0,
            "recall": round(tp/len(ground_truth), 4) if ground_truth else 0,
            "exact_match": predicted == ground_truth}
