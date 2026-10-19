"""GUI-facing experiment service. No direct dependency on Qt or user files in reports.

Jev/mocks supply pseudo-labels; deterministic rules and ground truth are separate.
No automatic production, no real customer data storage, no approved profiles.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .cli import _ground_truth, _write_csv
from .jev import JevClassifier, MockClassifier
from .openrouter import OpenRouterClassifier
from .pdftext import PageText, extract_pages, select_sample
from .rules import Candidate, evaluate, make_index, mine


def parse_page_ranges(value: str, total: int) -> list[int]:
    """Parse 1-based ranges e.g. '1-12, 15, 18-20'; reject invalid/duplicate pages."""
    if not value.strip():
        raise ValueError("Enter page ranges, for example 1-20 or 1-10,30-39.")
    positions: list[int] = []
    seen: set[int] = set()
    for token in value.split(","):
        pair = token.strip().split("-")
        if len(pair) > 2 or not all(x.strip().isdigit() for x in pair):
            raise ValueError("Invalid range. Use page numbers such as 1-20,25,30-33.")
        a = int(pair[0])
        b = int(pair[-1])
        if not 1 <= a <= b <= total:
            raise ValueError(f"Page range {token.strip()} outside 1-{total}.")
        if len(positions) + b - a + 1 > 200:
            raise ValueError("Prototype samples are limited to 200 pages per experiment.")
        for number in range(a, b + 1):
            if number in seen:
                raise ValueError(f"Page {number} was selected more than once.")
            seen.add(number)
            positions.append(number)
    if len(positions) < 3:
        raise ValueError("Select at least three source pages.")
    return sorted(positions)


@dataclass(frozen=True)
class Request:
    pdf: str
    backend: str = "mock"
    sampling: str = "first"
    sample_count: int = 20
    custom_ranges: str = ""
    ground_truth: str = ""
    allow_external_text: bool = False


@dataclass
class Experiment:
    request: Request
    page_count: int
    sampled: list[int]
    probabilities: dict[int, float]
    candidates: list[Candidate]
    selected_rank: int
    groups: list[tuple[int, int]]
    starts: list[int]
    ground_truth: dict | None

    @property
    def selected(self) -> Candidate:
        return self.candidates[self.selected_rank]

    def report(self) -> dict:
        return {
            "status": "EXPERIMENTAL_NOT_PRODUCTION_APPROVED",
            "backend": self.request.backend,
            "pdf_page_count": self.page_count,
            "sample_count": len(self.sampled),
            "pseudo_labels_are_not_ground_truth": True,
            "selected_candidate_rank": self.selected_rank + 1,
            "selected_rule": self.selected.to_dict(),
            "candidate_envelopes": len(self.groups),
            "candidate_start_pages": self.starts,
            "independent_ground_truth": self.ground_truth,
            "warning": "No mailpiece production, validation or approval is implied.",
        }


def run_experiment(request: Request, progress: Callable[[str], None] | None = None,
                   classifier=None) -> Experiment:
    def stage(s: str):
        if progress:
            progress(s)

    if request.backend not in {"mock", "jev", "openrouter"}:
        raise ValueError("Unknown decision backend.")
    if request.backend != "mock" and not request.allow_external_text:
        raise ValueError("Explicit external text upload approval is required.")
    if request.sampling not in {"first", "distributed", "custom"}:
        raise ValueError("Unsupported sampling method.")

    stage("Reading embedded PDF text and coordinates...")
    pages: list[PageText] = extract_pages(request.pdf)
    if request.sampling == "custom":
        selected = parse_page_ranges(request.custom_ranges, len(pages))
        sample = [pages[number - 1] for number in selected]
    else:
        if not 3 <= request.sample_count <= 200:
            raise ValueError("Sample size must be between 3 and 200.")
        sample = select_sample(pages, request.sample_count, request.sampling)
    if classifier is None:
        classifier = (MockClassifier() if request.backend == "mock" else
                      JevClassifier(allow_external_text=True) if request.backend == "jev" else
                      OpenRouterClassifier(allow_external_text=True))
    stage(f"Classifying {len(sample)} sample pages with {request.backend.upper()}...")
    probabilities = classifier.classify(sample)
    if set(probabilities) != {p.number for p in sample}:
        raise ValueError("Classifier returned an incomplete or invalid set of page results.")

    stage("Distilling interpretable first-page rule candidates...")
    candidates = mine(sample, probabilities)
    if not candidates:
        raise ValueError("No usable candidate rule. Try a more representative sample.")
    stage(f"Scanning all {len(pages):,} pages using candidate rules...")
    # The highest-scored candidate may fail to recognize page 1. Use the first
    # viable one, not a guessed/silently inserted first-page boundary.
    selected_rank = None
    groups: list[tuple[int, int]] = []
    starts: list[int] = []
    for i, candidate in enumerate(candidates[:12]):
        try:
            groups, starts = make_index(pages, candidate.rule)
            selected_rank = i
            break
        except ValueError:
            continue
    if selected_rank is None:
        raise ValueError("None of the top 12 rule candidates correctly recognizes source page 1.")
    stage("Checking independent ground truth (if supplied)...")
    truth = _ground_truth(Path(request.ground_truth), len(pages)) if request.ground_truth else None
    comparison = evaluate(starts, truth) if truth is not None else None
    stage("Candidate report ready. Not approved for production.")
    return Experiment(request, len(pages), [p.number for p in sample], probabilities,
                      candidates, selected_rank, groups, starts, comparison)


def export_experiment(result: Experiment, directory: str | Path) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    origin = {"mock": "MOCK_HEURISTIC", "jev": "JEV_API_UNCALIBRATED",
              "openrouter": "OPENROUTER_API_UNCALIBRATED"}[result.request.backend]
    _write_csv(path / "sample_decisions.csv",
               ["SourcePage", "FirstPageProbability", "Origin"],
               [[n, result.probabilities[n], origin] for n in result.sampled])
    _write_csv(path / "rule_candidates.csv",
               ["Rank", "Kind", "Term", "RegionMM", "TP", "FP", "FN", "TN", "Precision", "Recall"],
               [[i, c.rule.kind, c.rule.term or "", str(c.rule.region_mm or ""), c.tp, c.fp,
                 c.fn, c.tn, round(c.precision, 4), round(c.recall, 4)]
                for i, c in enumerate(result.candidates, 1)])
    _write_csv(path / "candidate_index.csv",
               ["Envelope", "StartPage", "EndPage", "PageCount", "ReviewStatus"],
               [[i, start, end, end-start+1, "UNVERIFIED_EXPERIMENTAL"]
                for i, (start, end) in enumerate(result.groups, 1)])
    (path / "candidate_detection_config.json").write_text(
        json.dumps(result.selected.rule.to_detection_config(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (path / "report.json").write_text(
        json.dumps(result.report(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
