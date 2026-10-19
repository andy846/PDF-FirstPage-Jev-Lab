"""Local rule discovery regression tests (synthetic text and geometry only)."""
import csv
from dataclasses import replace

import pytest

from firstpage_jev_lab.demo import make_demo
from firstpage_jev_lab.gui_service import Request, export_experiment, run_experiment
from firstpage_jev_lab.pdftext import PageText, TextLine
from firstpage_jev_lab.rules import Rule, _safe_phrases, matches, mine


def _page(number, first, *, continuation_salutation=False):
    lines = [
        TextLine("MEMBER STATEMENT", (10, 10, 95, 6)),
        TextLine("Account Number: 12345678", (16, 95, 100, 7)),
    ]
    if first:
        lines += [
            TextLine("Dear Member", (22, 43, 55, 7)),
            TextLine("Statement Date: [SYNTHETIC]", (22, 56, 100, 7)),
        ]
    elif continuation_salutation:
        # The exact same words, but they occur near the bottom of continuation pages.
        lines.append(TextLine("Dear Member", (22, 202, 55, 7)))
    return PageText(number, "\n".join(line.text for line in lines), tuple(lines), 210, 297)


def _sample():
    starts = {1, 4, 7, 10}
    pages = [_page(i, i in starts, continuation_salutation=i in (3, 6, 12))
             for i in range(1, 13)]
    probs = {p.number: .97 if p.number in starts else .03 for p in pages}
    return pages, probs


def test_spatial_marker_disambiguates_same_words_at_different_positions():
    pages, probs = _sample()
    candidates = mine(pages, probs)
    assert candidates
    best = candidates[0]
    assert best.rule.kind == "first_text"
    assert best.rule.region_mm is not None
    assert best.rule.term == "Dear Member"
    assert best.fp == 0 and best.fn == 0
    assert best.training["fp"] == 0
    assert best.pseudo_holdout["assessed_pages"] == 2
    assert best.pseudo_holdout["fp"] == best.pseudo_holdout["fn"] == 0
    assert all(matches(p, best.rule) == (p.number in {1, 4, 7, 10}) for p in pages)


def test_withheld_pages_cannot_drive_rule_mining_or_ranking():
    pages, probs = _sample()
    before = mine(pages, probs)
    # Last continuation is withheld. Move a duplicate marker into the learned
    # region on that page: training features and rank must NOT change.
    changed = [replace(p, lines=p.lines + (TextLine("Dear Member", (22, 43, 55, 7)),))
               if p.number == 12 else p for p in pages]
    after = mine(changed, probs)
    assert [c.rule.as_dict() for c in before] == [c.rule.as_dict() for c in after]
    assert [c.training for c in before] == [c.training for c in after]
    assert after[0].pseudo_holdout["fp"] == before[0].pseudo_holdout["fp"] + 1
    assert after[0].fp == before[0].fp + 1


def test_page_counter_narrow_region_beats_conflicting_footer_counter():
    pages = []
    starts = {1, 4, 7}
    for number in range(1, 10):
        current = (number - 1) % 3 + 1
        lines = (
            TextLine(f"Page {current} of 3", (150, 15, 35, 7)),
            TextLine("Page 1 of 20", (150, 272, 35, 7)),  # a misleading footer
        )
        pages.append(PageText(number, "\n".join(x.text for x in lines),
                              lines, 210, 297))
    probs = {page.number: .97 if page.number in starts else .03 for page in pages}
    candidates = mine(pages, probs)
    winners = [c for c in candidates if c.rule.kind == "page_number"
               and c.rule.region_mm is not None and c.fp == 0 and c.fn == 0]
    assert winners
    assert all(matches(p, winners[0].rule) == (p.number in starts) for p in pages)
    assert not matches(pages[0], Rule("page_number"))  # ambiguous full-page counters


def test_rejects_unconfident_and_invalid_pseudo_labels():
    pages, probs = _sample()
    with pytest.raises(ValueError, match="probabilities"):
        mine(pages, {**probs, 2: float("nan")})
    with pytest.raises(ValueError, match="IDs"):
        mine(pages, {1: .97})
    with pytest.raises(ValueError, match="thresholds"):
        mine(pages, probs, positive_min=.1, negative_max=.9)
    with pytest.raises(ValueError, match="confident"):
        mine(pages, {p.number: .5 for p in pages})


def test_pseudo_holdout_only_when_enough_samples():
    pages, probs = _sample()
    four = [pages[i-1] for i in (1, 2, 4, 5)]
    candidates = mine(four, {p.number: probs[p.number] for p in four})
    assert candidates
    assert candidates[0].pseudo_holdout is None


def test_candidate_text_terms_do_not_auto_include_personal_values():
    assert _safe_phrases("Account Number: 9876543") == set()
    assert _safe_phrases("Name: Example Customer") == set()
    assert _safe_phrases("ABC 123-45") == set()
    assert "Statement Date" in _safe_phrases("Statement Date: 2026-10-19")
    assert "Dear Member" in _safe_phrases("Dear Member")


def test_gui_service_reports_holdout_separately_from_independent_truth(tmp_path):
    pdf, expected = make_demo(tmp_path / "synthetic.pdf")
    truth_path = tmp_path / "truth.csv"
    with truth_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["StartPage"])
        w.writerows([[x] for x in expected])
    r = run_experiment(Request(str(pdf), ground_truth=str(truth_path)))
    assert r.ground_truth["exact_match"] is True
    assert r.selected.pseudo_holdout["assessed_pages"] == 2
    report = r.report()
    assert report["pseudo_holdout"]["assessed_pages"] == 2
    assert report["independent_ground_truth"]["exact_match"] is True
    assert "training" in report["selection_basis"]
    out = export_experiment(r, tmp_path / "export")
    with (out / "rule_candidates.csv").open(encoding="utf-8-sig", newline="") as f:
        row = next(csv.DictReader(f))
    assert "HoldoutFP" in row and "HoldoutFN" in row
