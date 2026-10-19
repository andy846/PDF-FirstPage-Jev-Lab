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
    training: dict | None = None
    pseudo_holdout: dict | None = None

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def score(self) -> tuple:
        # Rank ONLY against training pseudo-labels. Held-out labels must not
        # influence feature mining, candidate ranking, or rule choice.
        m = self.training or {"tp": self.tp, "fp": self.fp, "fn": self.fn}
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        return (fp == 0, tp > 0, precision, recall, tp, -fp,
                self.rule.region_mm is not None)

    def to_dict(self) -> dict:
        return {**self.rule.as_dict(), "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                "precision_on_sample": round(self.precision, 4),
                "recall_on_sample": round(self.recall, 4),
                "assessed_pages": self.assessed_pages,
                "training_pseudo_labels": self.training,
                "held_out_pseudo_labels": self.pseudo_holdout,
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
    if re.search(r"\b(?:name|address|tel|phone|email|account\s*(?:no\.?|number|id)|member\s*id)\b", line, re.I):
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


def _safe_phrases(line: str) -> set[str]:
    """Candidate literal strings, not raw customer values.

    Prefixes of labelled text are useful, but neither heuristics nor an LLM
    can prove that an arbitrary heading is safe to store in a profile.
    Candidate strings must be reviewed before any profile is approved.
    """
    term = _static_term(line)
    if not term:
        return set()
    terms = {term}
    # Long static headings may include different (non-numeric) suffixes.
    # Mine repeated two-to-four-word fragments rather than entire contents,
    # restricting fragments to reasonably descriptive label-like phrases.
    parts = term.split()
    if len(parts) >= 4:
        for length in (2, 3, 4):
            for i in range(len(parts)-length+1):
                phrase = " ".join(parts[i:i+length])
                if 9 <= len(phrase) <= 52:
                    terms.add(phrase)
    return terms


def _box_region(boxes: list[tuple[float, float, float, float]],
                page: PageText):
    if not boxes:
        return None
    # Prevent a poorly localized rule from expanding across most of the page.
    region = _region(boxes, page)
    if region and region[2] <= page.width_mm * .75 and region[3] <= page.height_mm * .18:
        return tuple(round(v, 2) for v in region)
    return None


def _page_number_boxes(page: PageText):
    for line in page.lines:
        for hit in PAGE_PATTERN.finditer(line.text):
            current, total = int(hit.group(1)), int(hit.group(2))
            if current == 1 and 1 <= total <= 10000:
                yield line.region_mm


def _scores(pages: list[PageText], rule: Rule, labels: dict[int, bool | None]) -> dict:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    failures = {"false_starts": [], "missed_starts": []}
    evaluated = 0
    for page in pages:
        expected = labels[page.number]
        if expected is None:
            continue
        evaluated += 1
        predicted = matches(page, rule)
        key = ("tp" if predicted else "fn") if expected else ("fp" if predicted else "tn")
        counts[key] += 1
        if key == "fp":
            failures["false_starts"].append(page.number)
        elif key == "fn":
            failures["missed_starts"].append(page.number)
    counts["assessed_pages"] = evaluated
    counts["precision"] = round(counts["tp"] / (counts["tp"] + counts["fp"]), 4) if counts["tp"] + counts["fp"] else 0.0
    counts["recall"] = round(counts["tp"] / (counts["tp"] + counts["fn"]), 4) if counts["tp"] + counts["fn"] else 0.0
    return {**counts, **failures}


def _training_holdout(pages: list[PageText], labels: dict[int, bool | None]
                      ) -> tuple[list[PageText], list[PageText]]:
    """Stratified, deterministic pseudo-label holdout, NEVER model ground truth."""
    positives = [p for p in pages if labels[p.number] is True]
    negatives = [p for p in pages if labels[p.number] is False]
    if len(positives) >= 3 and len(negatives) >= 2:
        # Withhold a first and a continuation page before generating features.
        withheld = {positives[-1].number, negatives[-1].number}
        return [p for p in pages if p.number not in withheld], [
            p for p in pages if p.number in withheld]
    return list(pages), []


def mine(sample_pages: list[PageText], probabilities: dict[int, float],
         *, positive_min: float = 0.8, negative_max: float = 0.2) -> list[Candidate]:
    """Distill deterministic literals and coordinates from noisy pseudo-labels.

    Rule features/ranks are computed ONLY on the training partition.
    The pseudo-holdout is reported but NEVER treated as independent truth.
    """
    if not 0 <= negative_max < positive_min <= 1:
        raise ValueError("Invalid positive/negative probability thresholds.")
    if len({p.number for p in sample_pages}) != len(sample_pages) or set(probabilities) != {p.number for p in sample_pages}:
        raise ValueError("Sample and predicted source-page IDs must match exactly.")
    if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1
           for v in probabilities.values()):
        raise ValueError("Classifier probabilities must be finite values between 0 and 1.")
    labels = {p.number: True if probabilities[p.number] >= positive_min else
              False if probabilities[p.number] <= negative_max else None
              for p in sample_pages}
    training_pages, holdout_pages = _training_holdout(sample_pages, labels)
    positives = [p for p in training_pages if labels[p.number] is True]
    negatives = [p for p in training_pages if labels[p.number] is False]
    if len(positives) < 2 or len(negatives) < 1:
        raise ValueError("Not enough confident first/continuation pages for rule discovery.")
    phrase_support = Counter()
    phrase_locations: dict[str, dict[tuple[int, int], list]] = defaultdict(lambda: defaultdict(list))
    page_boxes: dict[tuple[int, int], list] = defaultdict(list)
    counter_seen: dict[tuple[int, int], set[int]] = defaultdict(set)
    for page in positives:
        seen = set()
        for line in page.lines:
            for term in sorted(_safe_phrases(line.text)):
                if term not in seen:
                    phrase_support[term] += 1
                    seen.add(term)
                if term.casefold() in line.text.casefold():
                    x, y, w, h = line.region_mm
                    key = (round((x+w/2)/12), round((y+h/2)/9))
                    phrase_locations[term][key].append((page.number, line.region_mm))
        for box in _page_number_boxes(page):
            x, y, w, h = box
            key = (round((x+w/2)/12), round((y+h/2)/9))
            page_boxes[key].append(box)
            counter_seen[key].add(page.number)

    rules: list[Rule] = [Rule("page_number")]
    for position, ids in counter_seen.items():
        if len(ids) >= 2:
            region = _box_region(page_boxes[position], positives[0])
            if region:
                rules.append(Rule("page_number", region_mm=region))
    for term, support in phrase_support.most_common(140):
        if support < 2:
            continue
        rules.append(Rule("first_text", term))
        for positions in phrase_locations[term].values():
            if len({page_id for page_id, _ in positions}) < 2:
                continue
            region = _box_region([box for _, box in positions], positives[0])
            if region:
                rules.append(Rule("first_text", term, region))

    candidates: list[Candidate] = []
    fingerprints = set()
    for rule in rules:
        signature = (rule.kind, rule.term, rule.region_mm)
        if signature in fingerprints:
            continue
        fingerprints.add(signature)
        train = _scores(training_pages, rule, labels)
        if train["tp"] < 2:
            continue
        # Evaluate all pseudo-labelled pages for existing CSV/test compatibility.
        full = _scores(sample_pages, rule, labels)
        holdout = _scores(holdout_pages, rule, labels) if holdout_pages else None
        candidates.append(Candidate(rule, full["tp"], full["fp"], full["fn"],
                                    full["tn"], full["assessed_pages"],
                                    training=train, pseudo_holdout=holdout))
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
