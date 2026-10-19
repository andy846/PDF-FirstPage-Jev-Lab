# PDF FirstPage Jev Lab

**Independent experimental prototype — NOT production-ready.** This repository does not change PDFDocuEdit Pro and must not be used to release actual customer mailpieces.

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

The adapter calls the TypeSafe System One API and batches a noul question for each sampled page. **The live Jev integration has NOT been tested with a real Jev account.** The request format and returned decisions must still be verified. API calls can incur costs.

**Never submit real eMPF, bank or other confidential customer PDF text without explicit security and contractual approval.** The simple demonstration redaction is best-effort only, not reliable DLP. Keys, real PDFs and results must remain outside GitHub.

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
