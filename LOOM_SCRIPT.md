# 2-3 minute Loom outline (camera on)

0:00-0:25 Introduce the business objective: predict posted freight rates from shipment characteristics.
0:25-0:55 Show training data shape, feature groups, and exact missing-value counts from reports/metrics.json.
0:55-1:25 Explain the chronological split and why random splitting can exaggerate future performance.
1:25-1:55 Show both CatBoost holdout scores in the generated PDF and explain why December needs a separate model.
1:55-2:25 Open src/pipeline.py: feature preparation, fitting, prediction export; then show score.py output and December chart.
2:25-2:45 Summarize limitations, hidden validation, and reproducibility. Speak naturally and verify every result before recording.
