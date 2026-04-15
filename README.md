# warranty-prediction-system_V3

## EDA module

Run the complete EDA workflow:

```bash
python -m pip install -r requirements.txt
python -m eda.run_eda --input-file warranty_dataset.xlsx --sheet-name Clean_merge_data --output-dir outputs
```

Outputs are generated in:
- `outputs/eda_plots/`
- `outputs/eda_summary_statistics.json`
- `outputs/eda_insights.txt`
- `outputs/eda_report.html`
