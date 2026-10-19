# PDF FirstPage Jev Lab

**Independent experimental prototype — NOT production-ready.** This repository does not change PDFDocuEdit Pro and must not be used to release actual customer mailpieces.

## Desktop GUI (PySide6)

A standalone desktop window now provides **Open PDF**, **Create synthetic demo**, first/distributed/custom samples, Mock/Jev selection, per-page first-page probabilities, candidate rules, envelope index, local source-page preview, optional independent ground-truth comparison and CSV/JSON export. The GUI runs analysis on a background worker. Live Jev text transfer is **disabled by default** and requires explicit consent and confirmation.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[gui,test]'
firstpage-lab-gui
~~~

On Windows PowerShell, activate using .venv\Scripts\Activate.ps1 instead. Full setup and UI guide: [docs/GUI.md](docs/GUI.md).

**Recommended first test:** Click Create 20-page demo, then Analyze PDF in Mock mode. Mock classification is NOT Jev accuracy evidence. The external Jev backends have only been smoke-tested on the synthetic fixture; real customer documents remain unverified.

## Goal

Test whether TypeSafe **Jev** can classify likely **first pages versus continuation pages** from representative pages of a PDF containing real selectable text, and whether deterministic Python code can distill those noisy classifications into reusable first-page detection rules.

1. **PyMuPDF** extracts PDF text and coordinates.
2. **Jev** (optional API) estimates first-page likelihood for sampled pages.
3. A **Rule Miner** compares likely first/continuation pages and proposes literal-marker or page-counter rules.
4. A local deterministic scanner runs the chosen rule over the entire PDF to generate a candidate envelope index.
5. Optional independently verified ground truth measures false splits and missed starts.

Jev does not generate a detection rule itself. Pseudo-labels are not proof of correct grouping. No automatic Profile Library modification, barcode rendering or mailpiece production is included.

## Setup

Python 3.10+ on macOS, Windows or Linux.

~~~sh
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[test]'
python -m pytest -q
~~~

## Fully offline smoke test (no API key)

~~~sh
firstpage-lab make-demo --out samples/synthetic.pdf
firstpage-lab analyze samples/synthetic.pdf --backend mock --sample-count 20 --ground-truth samples/synthetic_truth.csv --out outputs/demo
~~~

The synthetic fixture contains 20 pages and seven letters. The mock classifier uses a simple hard-coded page-counter heuristic. Passing the mock test only checks the application workflow, not Jev accuracy.

## Live Jev mode (explicit opt-in)

An API key and organizational approval are required before transmitting any PDF text externally.

~~~sh
# macOS / Linux
export TYPESAFE_API_KEY='YOUR_PRIVATE_KEY'
export JEV_MODEL='jev-1.13.0'
firstpage-lab analyze samples/synthetic.pdf --backend jev --allow-external-text --sample-count 20 --out outputs/jev-demo
~~~

For Windows PowerShell, set the API key in a session environment variable. Do not store a real key in the repository.

The adapter batches a noul question for each sampled page. Live calls have been exercised against OpenCode-hosted `jev-1.13-free` and OpenRouter-hosted `typesafe/jev-1.14` on the synthetic fixture only; accuracy on real customer documents remains unverified. Requests are retried on 429/5xx responses, and reports record the model and provider actually served. API calls can incur costs.

### OpenCode-hosted Jev

OpenCode Console serves the same System One model at `https://opencode.ai/inference/systemone/v1/systemone`. Use an OpenCode Console service key (`oc_sk_...`) with the endpoint override:

~~~sh
export OPENCODE_API_KEY='oc_sk_...'
export JEV_ENDPOINT='https://opencode.ai/inference/systemone/v1/systemone'
export JEV_MODEL='jev-1.13-free'
firstpage-lab analyze samples/synthetic.pdf --backend jev --allow-external-text --sample-count 20 --out outputs/opencode-demo
~~~

`jev-1.13-free` is free while available; OpenCode refuses free models on networks it flags as abusive (VPN/datacenter IPs). The paid `jev-1.13` requires workspace credit.

When only `OPENCODE_API_KEY` is set, the endpoint and model default to the OpenCode gateway and `jev-1.13`. Set `JEV_ENDPOINT`/`JEV_MODEL` explicitly to override.

### OpenRouter-hosted Jev 1.14

The `openrouter` backend sends the same redacted System One payload through OpenRouter's alpha decisions endpoint (`typesafe/jev-1.14` by default, the real Jev decisions model — not the `jev-router` chat proxy that forwards to other LLMs):

~~~sh
export OPENROUTER_API_KEY='sk-or-v1-...'
firstpage-lab analyze samples/synthetic.pdf --backend openrouter --allow-external-text --sample-count 20 --out outputs/openrouter-demo
~~~

`OPENROUTER_MODEL` pins another TypeSafe decisions model such as `typesafe/jev-1.13`; `OPENROUTER_ENDPOINT` overrides the endpoint. Replies are uncalibrated pseudo-labels, never ground truth.

**Never submit real eMPF, bank or other confidential customer PDF text without explicit security and contractual approval.** The simple demonstration redaction is best-effort only, not reliable DLP. Keys, real PDFs and results must remain outside GitHub.

## Rule Miner v2: spatial features and pseudo-holdout checks

The local miner now extracts repeated **static text markers**, short repeated phrases, and **position-clustered** rules. It also searches narrow page-counter regions for layouts with more than one printed counter. It does not use OCR or a second language model.

Only confident Jev/Mock first and continuation predictions participate in discovery. With at least three candidate first pages and two continuation pages, one of each is **withheld before mining and ranking**, then evaluated later. The training and held-out pseudo-label metrics are **both comparisons against model pseudo-labels**, not verified envelope boundaries. Rule selection never inspects withheld pages.

For verified accuracy, provide an independent StartPage CSV and inspect the independent_ground_truth section of report.json. A holdout disagreement is flagged in the GUI. Never automatically approve a detected profile. All text-derived candidate strings require manual inspection before saving, to avoid capturing customer details.

The left sidebar now handles long dropdown options: shortened visible names, tooltip detail, width-constrained selectors and a wrapped external-transfer explanation. All external backends require explicit consent plus final confirmation.

## Sampling

~~~sh
firstpage-lab analyze my.pdf --backend mock --sampling first --sample-count 20 --out outputs/first20
firstpage-lab analyze my.pdf --backend mock --sampling distributed --sample-count 20 --out outputs/distributed20
~~~

The first 20 pages may miss later letter templates. Distributed sampling improves coverage but does not guarantee full accuracy. The prototype holds PDF text in memory; future large-volume usage should use a SQLite index.

## Reports

- **sample_decisions.csv**: predicted first-page probabilities and MOCK/JEV origin
- **rule_candidates.csv**: candidates and precision/recall against model pseudo-labels only
- **candidate_detection_config.json**: proposed DetectionConfig-style rule, not an approved Profile
- **candidate_index.csv**: proposed StartPage/EndPage with UNVERIFIED_EXPERIMENTAL status
- **report.json**: summary and independent ground-truth evaluation, if provided

For a fully verified PDF, provide ascending unique StartPage values as ground-truth CSV:

~~~csv
StartPage
1
4
6
10
13
15
18
~~~

Boundary evaluation must compare exact starts, not merely envelope counts.

## Limitations and testing

- Text PDFs only. No OCR, VLM or visual symbol detection.
- Single candidate first-text/page-counter rule; no multi-template resolver yet.
- Predictions may be wrong or uncalibrated. Independently validated boundaries and operator review are required.
- No authorized production output, no self-learning and no Profile Library overwriting.
- Jev API calls require explicit opt-in.
- No actual physical inserter validation.

GitHub Actions runs synthetic offline tests on push. Real customer data, API keys, PDFs and output reports are intentionally excluded by gitignore. After installation, run:

~~~sh
python -m pytest -q
~~~
