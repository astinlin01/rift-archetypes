"""
Merge dashboard/template.html with the exported dashboard JSON payload into a
single self-contained HTML file, ready to publish as an Artifact.

Run from repo root (after export_dashboard_data.py):
    python src/build_dashboard.py
"""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the self-contained dashboard HTML file.")
    parser.add_argument("--template-path", default="dashboard/template.html")
    parser.add_argument("--data-path", default="data/processed/dashboard_data.json")
    parser.add_argument("--output-path", default="dashboard/scouting_report.html")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if not os.path.exists(args.template_path):
        raise FileNotFoundError(f"Template not found: {args.template_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dashboard data not found: {args.data_path}. Run export_dashboard_data.py first.")

    with open(args.template_path, "r", encoding="utf-8") as infile:
        template = infile.read()
    with open(args.data_path, "r", encoding="utf-8") as infile:
        data = infile.read()

    if "__DATA_PLACEHOLDER__" not in template:
        raise RuntimeError("Template is missing the __DATA_PLACEHOLDER__ marker.")

    output = template.replace("__DATA_PLACEHOLDER__", data)

    parent = os.path.dirname(args.output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as outfile:
        outfile.write(output)

    print(f"Dashboard HTML written to: {args.output_path}")


if __name__ == "__main__":
    run(parse_args())
