"""Run the complete Week 3 customer segmentation and model evaluation project."""

from __future__ import annotations

from pathlib import Path

from src.clustering import run_clustering
from src.evaluation import run_classification, write_final_report
from src.preprocessing import inspect_dataset, load_customer_data

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    customers, customer_source = load_customer_data(data_dir)
    print(f"Customer clustering dataset source: {customer_source}")
    customer_checks = inspect_dataset(customers, "Customer clustering dataset")
    clustering = run_clustering(customers, figures_dir, reports_dir)
    classification = run_classification(figures_dir, reports_dir)
    write_final_report(
        reports_dir / "model_evaluation_report.md",
        customer_source,
        customer_checks,
        clustering,
        classification,
    )

    print(f"\nFinal report: {reports_dir / 'model_evaluation_report.md'}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()