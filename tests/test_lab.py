"""All tests are synthetic and offline; no real Jev API calls."""
import csv
import json
from pathlib import Path

import pytest

from firstpage_jev_lab.demo import make_demo
from firstpage_jev_lab.pdftext import extract_pages, select_sample
from firstpage_jev_lab.jev import (JevClassifier, MockClassifier, OFFICIAL_ENDPOINT,
                                   OPENCODE_ENDPOINT, USER_AGENT, redact, _parse_answer)
from firstpage_jev_lab.openrouter import OpenRouterClassifier
from firstpage_jev_lab.rules import Rule, evaluate, make_index, matches, mine
from firstpage_jev_lab.cli import main


@pytest.fixture
def fixture(tmp_path):
    pdf, starts = make_demo(tmp_path / "synthetic.pdf")
    return pdf, starts, extract_pages(pdf)


def test_selectable_text_and_coordinates(fixture):
    _, starts, pages = fixture
    assert len(pages) == 20
    assert starts == [1, 4, 6, 10, 13, 15, 18]
    assert "MEMBER STATEMENT" in pages[0].text
    assert pages[0].width_mm > 200
    assert all(p.lines for p in pages)


def test_sampling_strategies(fixture):
    *_, pages = fixture
    assert [p.number for p in select_sample(pages, 5, "first")] == [1, 2, 3, 4, 5]
    sample = select_sample(pages, 5, "distributed")
    assert sample[0].number == 1 and sample[-1].number == 20
    assert len({p.number for p in sample}) == 5


def test_offline_mock_and_rule_miner(fixture):
    _, expected, pages = fixture
    sample = select_sample(pages, 20, "first")
    probabilities = MockClassifier().classify(sample)
    assert {p for p, v in probabilities.items() if v >= 0.8} == set(expected)
    candidates = mine(sample, probabilities)
    assert candidates
    assert candidates[0].fp == 0 and candidates[0].tp == len(expected)
    groups, starts = make_index(pages, candidates[0].rule)
    assert starts == expected
    assert groups == [(1, 3), (4, 5), (6, 9), (10, 12), (13, 14), (15, 17), (18, 20)]
    assert evaluate(starts, set(expected))["exact_match"]


def test_repeated_marker_continuation_not_selected_if_bad(fixture):
    _, expected, pages = fixture
    candidates = mine(pages, MockClassifier().classify(pages))
    problematic = [c for c in candidates if c.rule.kind == "first_text" and c.rule.term == "Statement Date"]
    assert problematic and all(c.fp >= 1 for c in problematic)
    assert candidates[0].fp == 0


def test_rule_export_shape(fixture):
    _, _, pages = fixture
    candidates = mine(pages, MockClassifier().classify(pages))
    config = candidates[0].rule.to_detection_config()
    assert config["version"] == 2
    assert config["rules"][0]["kind"] in {"first_text", "page_number"}
    assert config["profile_name"] == "CANDIDATE_ONLY"


def test_missing_first_page_rejected(fixture):
    _, _, pages = fixture
    with pytest.raises(ValueError, match="First source page"):
        make_index(pages, Rule("first_text", "IMPOSSIBLE_MARKER"))


def test_jev_requires_explicit_external_upload(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="permission"):
        JevClassifier(key="unused")
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        JevClassifier(key="", allow_external_text=True)


def test_opencode_gateway_key_endpoint_and_model_resolution(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_API_KEY", "SYNTHETIC_TEST_ONLY")
    monkeypatch.setenv("JEV_ENDPOINT", OPENCODE_ENDPOINT)
    monkeypatch.setenv("JEV_MODEL", "jev-1.13-free")
    client = JevClassifier(allow_external_text=True, transport=lambda payload: {"answers": {}})
    assert (client.key, client.model, client.endpoint) == ("SYNTHETIC_TEST_ONLY", "jev-1.13-free",
                                                           OPENCODE_ENDPOINT)
    monkeypatch.delenv("JEV_ENDPOINT")
    monkeypatch.delenv("JEV_MODEL")
    assert JevClassifier(key="k", allow_external_text=True).endpoint == OFFICIAL_ENDPOINT


def test_jev_request_and_response_contract_offline(fixture, monkeypatch):
    monkeypatch.delenv("JEV_MODEL", raising=False)
    monkeypatch.delenv("JEV_ENDPOINT", raising=False)
    _, _, pages = fixture
    requests = []
    def fake_api(payload):
        requests.append(payload)
        return {"answers": {k: {"type": "noul", "noul": .93 if int(k[5:]) == 1 else .04}
                            for k in payload["questions"]}}
    client = JevClassifier(key="fake-key", allow_external_text=True, transport=fake_api)
    answer = client.classify(pages[:3], batch_size=2)
    assert answer == {1: .93, 2: .04, 3: .04}
    assert len(requests) == 2
    assert requests[0]["questions"]["page_1"]["type"] == "noul"
    assert requests[0]["model"] == "jev-1.13.0"
    assert requests[0]["state"]["pages"][0]["page"] == 1


def test_http_sends_identifying_user_agent(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return b'{"answers": {}}'

    def fake_urlopen(request, timeout=0):
        captured["request"] = request
        return FakeResponse()

    monkeypatch.setattr("firstpage_jev_lab.jev.urlopen", fake_urlopen)
    client = JevClassifier(key="k", allow_external_text=True)
    assert client._http({"model": client.model, "questions": {}}) == {"answers": {}}
    assert captured["request"].get_header("User-agent") == USER_AGENT


def test_openrouter_classifier_offline_contract(fixture, monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_ENDPOINT", raising=False)
    _, _, pages = fixture
    requests = []
    def fake_api(payload):
        requests.append(payload)
        return {"answers": {k: {"type": "noul", "noul": .91 if int(k[5:]) == 1 else .06}
                            for k in payload["questions"]}}
    client = OpenRouterClassifier(key="fake-key", allow_external_text=True, transport=fake_api)
    answer = client.classify(pages[:3], batch_size=2)
    assert answer == {1: .91, 2: .06, 3: .06}
    assert len(requests) == 2
    assert requests[0]["model"] == "typesafe/jev-1.14"
    assert requests[0]["questions"]["page_1"]["type"] == "noul"
    assert requests[0]["state"]["pages"][0]["page"] == 1
    assert client.endpoint.endswith("/api/alpha/decisions")


def test_openrouter_requires_permission_and_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="permission"):
        OpenRouterClassifier(key="unused")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterClassifier(key="", allow_external_text=True)


def test_openrouter_missing_answers_rejected(fixture):
    _, _, pages = fixture
    client = OpenRouterClassifier(key="fake-key", allow_external_text=True,
                                  transport=lambda payload: {"answers": None})
    with pytest.raises(ValueError, match="answers"):
        client.classify(pages[:2], batch_size=2)
    client = OpenRouterClassifier(key="fake-key", allow_external_text=True,
                                  transport=lambda payload: {"answers": {"page_7": {"type": "noul", "noul": .5}}})
    with pytest.raises(ValueError, match="page 1"):
        client.classify(pages[:2], batch_size=2)


def test_malformed_jev_output_rejected():
    for item in [None, {}, {"noul": -1}, {"noul": float("nan")}, {"noul": "0.5"},
                 {"noul": 1.1}, {"type": "choice", "noul": .7}]:
        with pytest.raises(ValueError):
            _parse_answer(item)


def test_redaction_best_effort():
    s = redact("Statement Date: 2026-10-19, Account: 123456789 and person@example.com")
    assert "2026-10-19" not in s
    assert "123456789" not in s
    assert "person@example.com" not in s
    assert "Statement Date:" in s


def test_cli_offline_end_to_end(tmp_path):
    pdf = tmp_path / "synthetic.pdf"
    main(["make-demo", "--out", str(pdf)])
    truth = pdf.with_name(pdf.stem + "_truth.csv")
    out = tmp_path / "output"
    main(["analyze", str(pdf), "--backend", "mock", "--out", str(out),
          "--ground-truth", str(truth)])
    assert (out / "candidate_index.csv").exists()
    assert (out / "candidate_detection_config.json").exists()
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["independent_ground_truth"]["exact_match"] is True
    assert report["status"] == "EXPERIMENTAL_NOT_PRODUCTION_APPROVED"
    with (out / "candidate_index.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["StartPage"] == "1"
    assert rows[0]["ReviewStatus"] == "UNVERIFIED_EXPERIMENTAL"


def test_cli_rejects_live_upload_by_default(fixture, tmp_path):
    pdf, *_ = fixture
    with pytest.raises(SystemExit) as caught:
        main(["analyze", str(pdf), "--backend", "jev", "--out", str(tmp_path / "out")])
    assert caught.value.code == 2
    assert not (tmp_path / "out").exists()


def test_cli_rejects_openrouter_upload_by_default(fixture, tmp_path):
    pdf, *_ = fixture
    with pytest.raises(SystemExit) as caught:
        main(["analyze", str(pdf), "--backend", "openrouter", "--out", str(tmp_path / "out")])
    assert caught.value.code == 2
    assert not (tmp_path / "out").exists()
