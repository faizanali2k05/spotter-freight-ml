# Spotter Freight Rate Prediction Assessment

Reproducible freight-rate regression with chronological holdout validation.

## Setup (Linux/macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Windows PowerShell: `.venv\\Scripts\\Activate.ps1`.

## Reproduce
```bash
python src/pipeline.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python src/make_report.py
```

## Files
- `data/train_test.csv`: 48,000 labeled loads; target `posted_rate`.
- `data/validation.csv`: 12,000 unseen loads.
- `validation_predictions.csv`: submission file, exact `load_id,predicted_rate` schema.
- `data/december_chart_inputs.csv`: 31 completed December inputs.
- `reports/metrics.json`: measured chronological holdout scores.
- `scorer_results/candidate_december.png`: required chart.
- `reports/assessment_report.pdf`: report.

## Method and limitations
Dates are kept intact across the train/holdout boundary. The *full* CatBoost model uses location, route, time, market index and quote signal for final validation. The separate *December-compatible* model uses only features provided for December plus date-derived fields. This prevents invented market/quote/location values. The December chart is a forecast scenario, not a validation against known December rates. Models are re-fit on the entire labeled development dataset after holdout evaluation. Final hidden-set metrics are calculated by Spotter, not by `score.py`.

## Submission checklist
GitHub URL, `validation_predictions.csv`, `reports/assessment_report.pdf`, 2-3 minute Loom recording with camera on. Review and explain the methodology in your own words before submission.
