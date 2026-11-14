"""
Merge dashboard/template.html, the exported dashboard JSON payload and the shared browser
math (dashboard/profile_math.js) into a single self-contained HTML file -- a static page
that can be hosted anywhere (GitHub Pages) or opened locally.

Writes docs/index.html, the GitHub Pages site. Run from repo root (after export_dashboard_data.py):
    python src/build_dashboard.py
"""

from __future__ import annotations

import argparse
import os

DATA_PLACEHOLDER = "__DATA_PLACEHOLDER__"
MATH_PLACEHOLDER = "/*__PROFILE_MATH__*/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the self-contained dashboard HTML file.")
    parser.add_argument("--template-path", default="dashboard/template.html")
    parser.add_argument("--data-path", default="data/processed/dashboard_data.json")
    parser.add_argument("--math-path", default="dashboard/profile_math.js")
    # docs/ is what GitHub Pages serves, so a rebuild is also the site update.
    parser.add_argument("--output-path", default="docs/index.html")
    return parser.parse_args()


def _read(path: str, hint: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. {hint}")
    with open(path, "r", encoding="utf-8") as infile:
        return infile.read()


def run(args: argparse.Namespace) -> None:
    template = _read(args.template_path, "")
    data = _read(args.data_path, "Run export_dashboard_data.py first.")
    math = _read(args.math_path, "")

    for placeholder in (DATA_PLACEHOLDER, MATH_PLACEHOLDER):
        if placeholder not in template:
            raise RuntimeError(f"Template is missing the {placeholder} marker.")

    # The HTML parser ends a <script> block at the first "</" + "script", even inside a
    # JSON string. "<\/" means the same thing to JSON.parse, so escape it.
    safe_data = data.replace("</", "<\\/")
    output = template.replace(DATA_PLACEHOLDER, safe_data).replace(MATH_PLACEHOLDER, math)

    parent = os.path.dirname(args.output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as outfile:
        outfile.write(output)

    print(f"Dashboard HTML written to: {args.output_path} ({os.path.getsize(args.output_path) / 1024:.0f} KB)")


if __name__ == "__main__":
    run(parse_args())
