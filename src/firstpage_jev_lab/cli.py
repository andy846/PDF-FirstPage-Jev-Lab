"""Command line tool: explicitly experimental and never production-approved."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from .demo import make_demo
from .jev import JevClassifier, MockClassifier
from .openrouter import OpenRouterClassifier
from .pdftext import extract_pages, select_sample
from .rules import evaluate, make_index, mine


def _write_csv(path: Path, header: list[str], rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _ground_truth(path: Path | None, page_count: int) -> set[int] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "StartPage" not in (reader.fieldnames or []):
            raise ValueError("Ground truth CSV requires a StartPage column.")
        starts = [int(r["StartPage"]) for r in reader]
    if not starts or starts[0] != 1 or starts != sorted(set(starts)) or starts[-1] > page_count:
        raise ValueError("Ground truth StartPage must be unique, sorted, begin at 1 and be in range.")
    return set(starts)


def analyze(args):
    if args.backend != "mock" and not args.allow_external_text:
        raise ValueError("External upload is disabled by default. Pass --allow-external-text only for approved/synthetic documents.")
    pages = extract_pages(args.pdf)
    sample = select_sample(pages, args.sample_count, args.sampling)
    classifier = (MockClassifier() if args.backend == "mock" else
                  JevClassifier(key=os.getenv("TYPESAFE_API_KEY"),
                                model=os.getenv("JEV_MODEL", "jev-1.13.0"),
                                allow_external_text=args.allow_external_text) if args.backend == "jev" else
                  OpenRouterClassifier(key=os.getenv("OPENROUTER_API_KEY"),
                                       model=os.getenv("OPENROUTER_MODEL", "typesafe/jev-1.14"),
                                       allow_external_text=args.allow_external_text))
    probabilities = classifier.classify(sample)
    candidates = mine(sample, probabilities)
    if not candidates:
        raise ValueError("No usable rule candidates. Need more representative labeled samples.")
    origin = {"mock": "MOCK_HEURISTIC", "jev": "JEV_API", "openrouter": "OPENROUTER_API"}[args.backend]
    model = {"mock": "MOCK_ONLY",
             "jev": os.getenv("JEV_MODEL", "jev-1.13.0"),
             "openrouter": os.getenv("OPENROUTER_MODEL", "typesafe/jev-1.14")}[args.backend]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "sample_decisions.csv", ["SourcePage", "FirstPageProbability", "Origin"],
               [[p.number, probabilities[p.number], origin] for p in sample])
    _write_csv(out / "rule_candidates.csv", ["Rank", "Kind", "Term", "RegionMM", "TP", "FP", "FN", "TN", "Precision", "Recall"],
               [[i, c.rule.kind, c.rule.term or "", str(c.rule.region_mm or ""), c.tp, c.fp, c.fn, c.tn,
                 round(c.precision, 4), round(c.recall, 4)] for i, c in enumerate(candidates, 1)])
    best = candidates[0]
    groups, starts = make_index(pages, best.rule)
    _write_csv(out / "candidate_index.csv", ["Envelope", "StartPage", "EndPage", "PageCount", "ReviewStatus"],
               [[i, a, b, b-a+1, "UNVERIFIED_EXPERIMENTAL"] for i, (a, b) in enumerate(groups, 1)])
    with (out / "candidate_detection_config.json").open("w", encoding="utf-8") as f:
        json.dump(best.rule.to_detection_config(), f, indent=2, ensure_ascii=False)
    truth = _ground_truth(Path(args.ground_truth) if args.ground_truth else None, len(pages))
    report = {"status": "EXPERIMENTAL_NOT_PRODUCTION_APPROVED", "backend": args.backend,
              "model": model,
              "pdf_page_count": len(pages), "sample_count": len(sample),
              "pseudo_label_warning": "Model/mocked probabilities are NOT independently verified ground truth.",
              "candidate_rule": best.to_dict(), "candidate_envelope_count": len(groups),
              "candidate_source_start_pages": starts,
              "independent_ground_truth": evaluate(starts, truth) if truth is not None else None,
              "limitations": ["Only PDFs with embedded selectable text", "Only first_text/page_number candidates",
                              "No production authorization", "Possible missed document types outside sample",
                              "External upload requires explicit opt-in; redaction is not guaranteed"]}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "experimental": True,
                      "best_rule": best.rule.as_dict(), "mailpieces": len(groups),
                      "ground_truth_exact": report["independent_ground_truth"]["exact_match"] if truth else None},
                     ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="firstpage-lab", description="Experimental only: Jev + PDF first-page rule distillation")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make-demo", help="Build synthetic text PDF + independent StartPage CSV")
    make.add_argument("--out", default="samples/synthetic.pdf")
    run = commands.add_parser("analyze", help="Classify sample, mine a rule, and scan entire PDF")
    run.add_argument("pdf")
    run.add_argument("--backend", choices=("mock", "jev", "openrouter"), default="mock")
    run.add_argument("--sampling", choices=("first", "distributed"), default="first")
    run.add_argument("--sample-count", type=int, default=20)
    run.add_argument("--out", default="outputs/trial")
    run.add_argument("--ground-truth", help="Independent CSV with StartPage column")
    run.add_argument("--allow-external-text", action="store_true",
                     help="Explicit approval for text upload to the external model API")
    args = parser.parse_args(argv)
    try:
        if args.command == "make-demo":
            path, starts = make_demo(args.out)
            truth = path.with_name(path.stem + "_truth.csv")
            _write_csv(truth, ["StartPage"], [[p] for p in starts])
            print(f"Synthetic fixture: {path}\nIndependent truth: {truth}")
        else:
            analyze(args)
    except (ValueError, RuntimeError, OSError) as err:
        parser.exit(status=2, message=f"Error: {err}\n")


if __name__ == "__main__":
    main()
