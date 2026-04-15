from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import ensure_output_dirs, load_dataset, save_json
    from .eda_data_overview import run_data_overview
    from .eda_relationships import run_relationship_analysis
    from .eda_risk_analysis import run_risk_analysis
    from .eda_summary_report import create_summary_report
    from .eda_temporal import run_temporal_analysis
    from .eda_univariate import run_univariate_analysis
    from .eda_visualizations import generate_visualizations
except ImportError:  # pragma: no cover
    from common import ensure_output_dirs, load_dataset, save_json
    from eda_data_overview import run_data_overview
    from eda_relationships import run_relationship_analysis
    from eda_risk_analysis import run_risk_analysis
    from eda_summary_report import create_summary_report
    from eda_temporal import run_temporal_analysis
    from eda_univariate import run_univariate_analysis
    from eda_visualizations import generate_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full EDA workflow for warranty dataset")
    parser.add_argument(
        "--input-file",
        default="warranty_dataset.xlsx",
        help="Path to Excel dataset file",
    )
    parser.add_argument("--sheet-name", default="Clean_merge_data", help="Excel sheet name")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for reports and plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = ensure_output_dirs(args.output_dir)
    df = load_dataset(args.input_file, sheet_name=args.sheet_name)

    results = {
        "data_overview": run_data_overview(df, output_paths["distributions"]),
        "univariate": run_univariate_analysis(df, output_paths["distributions"]),
        "relationships": run_relationship_analysis(df, output_paths["correlations"]),
        "temporal": run_temporal_analysis(df, output_paths["temporal"]),
        "risk_analysis": run_risk_analysis(df, output_paths["risk_analysis"]),
        "visualizations": generate_visualizations(df, output_paths["geographic"]),
    }

    summary = create_summary_report(results, output_paths["base"])
    results["summary"] = summary
    save_json(results, Path(args.output_dir) / "eda_summary_statistics.json")

    print(f"EDA completed. Outputs saved under: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
