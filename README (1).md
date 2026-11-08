# Kazakh QA: original vs machine-translated questions (Assignment 4)

**Research question.** How does the accuracy of one selected open-source LLM differ between original Kazakh and
machine-translated Kazakh question-answering datasets?

Data: [KazQAD](https://github.com/IS2AI/KazQAD) @ `ab8684a667109f2a7345f4f89ab3b123e8d78e1f` (CC BY-SA 4.0).
Original group = KazQAD test (UNT exam questions, Kazakh Wikipedia passages). Translated group = nq-translate-kk
(Natural Questions machine-translated into Kazakh). Frozen test set: `results_main/selected.json`,
100 items per group, SHA-256 `049831f61160d3657d4eed27548fcf3c5007ade588804982402142bae11b6eb5`.

All frozen decisions: `configs/protocol.json`.

## Structure
| Path | Content |
|---|---|
| `qa_compare.py` | data selection, Qwen open-book inference, EM / token-F1 scoring (Assignments 2–3) |
| `group_analysis.py` | lexical baseline, covariates and data-issue flags (Assignment 3) |
| `a4_experiment.py` | Assignment 4 runner: lexical, XLM-R, Qwen closed-book, analysis, kappa |
| `configs/protocol.json` | frozen protocol |
| `results_pilot/` | pilot selection (excluded from the test set) |
| `results_main/` | frozen test set; after the run: `pred_*.json`, `manifest_*.json`, `analysis/` |
| `A4_run_experiment.ipynb` | one-click Colab run |
| `tests/` | software test of the analysis pipeline |

## Reproduce the main results table
Google Colab, T4 GPU (≈20–30 min): open `A4_run_experiment.ipynb`, set `REPO_URL`, Runtime → Run all.

Or locally (GPU recommended):
```bash
pip install -r requirements.txt
python qa_compare.py self-test
sha256sum results_main/selected.json            # 049831f6...
python a4_experiment.py lexical
python a4_experiment.py xlmr
python qa_compare.py run --out results_main --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
python a4_experiment.py import-qwen
python a4_experiment.py qwen-closed
python a4_experiment.py analyse                 # -> results_main/analysis/main_results.csv, statistics.json, figures
# after two people code results_main/analysis/error_analysis_sheet.csv:
python a4_experiment.py kappa
```
Re-creating the frozen test set from the raw data (optional):
```bash
git clone https://github.com/IS2AI/KazQAD kazqad_source && git -C kazqad_source checkout ab8684a667109f2a7345f4f89ab3b123e8d78e1f
python qa_compare.py prepare --data kazqad_source --n 100 --out check --exclude-sample results_pilot/selected.json
sha256sum check/selected.json                   # must equal the hash above
```
Existing prediction files are never overwritten; archive them before re-running.

## Licence
Code: MIT. KazQAD excerpts: CC BY-SA 4.0 (IS2AI).
