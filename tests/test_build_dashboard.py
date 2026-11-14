"""
build_dashboard.py: merges the template, the exported data and profile_math.js into one
self-contained HTML file (what gets deployed).
"""

import argparse
import json

import pytest

from build_dashboard import run


def build(tmp_path, data: dict, template: str = None) -> str:
    template_path = tmp_path / "template.html"
    template_path.write_text(template or (
        "<!doctype html><html><body>"
        '<script id="dashboard-data" type="application/json">__DATA_PLACEHOLDER__</script>'
        "<script>/*__PROFILE_MATH__*/</script></body></html>"
    ), encoding="utf-8")
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(data), encoding="utf-8")
    math_path = tmp_path / "profile_math.js"
    math_path.write_text("var ProfileMath = {};", encoding="utf-8")
    output_path = tmp_path / "out" / "index.html"
    run(argparse.Namespace(template_path=str(template_path), data_path=str(data_path),
                           math_path=str(math_path), output_path=str(output_path)))
    return output_path.read_text(encoding="utf-8")


def test_inlines_data_and_math(tmp_path):
    html = build(tmp_path, {"season": 16})
    assert '"season": 16' in html
    assert "var ProfileMath = {};" in html
    assert "__DATA_PLACEHOLDER__" not in html and "__PROFILE_MATH__" not in html


def test_data_cannot_close_the_script_tag(tmp_path):
    # A "</script>" inside a JSON string would end the data block early and break the
    # page; "<\/" is the same string to JSON.parse but not to the HTML parser.
    html = build(tmp_path, {"name": "evil</script><b>"})
    assert "evil</script>" not in html
    assert json.loads(html.split('type="application/json">')[1].split("</script>")[0]) == {"name": "evil</script><b>"}


def test_missing_placeholder_fails(tmp_path):
    with pytest.raises(RuntimeError):
        build(tmp_path, {}, template="<html>no placeholders</html>")
