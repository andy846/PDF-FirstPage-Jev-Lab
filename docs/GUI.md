# PySide6 desktop GUI — PDF FirstPage Jev Lab

This is an **experimental research app**, not a mailpiece production program.

## Install and launch

Python 3.10+ required. Run in the repository folder.

### macOS

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[gui,test]'
firstpage-lab-gui
~~~

### Windows PowerShell

~~~powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e '.[gui,test]'
firstpage-lab-gui
~~~

You may also run: python -m firstpage_jev_lab.gui

## Step-by-step

1. Click **Create 20-page demo**. This creates a synthetic text PDF and known StartPage CSV in a machine temporary folder.
2. Keep **First N pages**, **20**, and **Mock / Fully offline**.
3. Click **Analyze PDF**. A background Qt worker performs analysis, avoiding a frozen GUI.
4. Examine **Page decisions**, **Discovered rules**, and **Envelope index**. Select a row and open the **PDF preview** tab to inspect the source page.
5. Check **VERIFICATION**. MATCH means the proposed exact boundaries match the *supplied independent ground-truth CSV*; UNVERIFIED means no ground truth was provided.
6. **Export CSV / JSON reports** into a directory you choose.

For your own **nonconfidential** text PDF, open it and optionally load an independently verified CSV with a StartPage column. Use first-N, distributed-N, or custom ranges such as 1-20,35-40. A contiguous first-20 sample can miss new templates later.

## Live TypeSafe Jev API

**Default mode is completely offline**. Opening a PDF never sends it to TypeSafe.

Set TYPESAFE_API_KEY in your shell before starting the GUI:

~~~sh
export TYPESAFE_API_KEY='YOUR_PRIVATE_KEY'
firstpage-lab-gui
~~~

Then choose **Jev / TypeSafe external API**. The GUI requires:

- An environment-variable API key, never displayed or persisted by the app.
- Explicit unchecked-by-default user consent for every new experiment.
- A final confirmation dialog before any transfer.

**Never use real confidential customer PDFs without explicit security and contractual approval.** Best-effort redaction does not guarantee removal of personal data. Jev request/response contract tests currently use fakes; actual TypeSafe API compatibility, model access, rate limits and cost are not yet verified.

## Limitations

- PDF must contain embedded selectable text. No OCR or visual feature learning.
- Jev suggests pseudo-labels for first/continuation pages; local code mines deterministic rules.
- Only a single top candidate is applied, with fallback if the first-ranked candidate fails page 1. Candidates are not production-approved.
- Reports and preview stay local unless Jev is explicitly enabled; model requests contain *redacted but potentially sensitive* PDF-derived text.
- No auto Profile Library update, no inserter test, no production printing.

## Headless tests

~~~bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
~~~

CI installs the GUI extras and tests offscreen using synthetic documents only.
