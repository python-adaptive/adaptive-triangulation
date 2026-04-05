from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def pyproject_python_versions() -> list[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    classifiers = data["project"]["classifiers"]
    return sorted(
        match.group(1)
        for classifier in classifiers
        if (match := re.fullmatch(r"Programming Language :: Python :: (3\.\d+)", classifier))
    )


def workflow_python_versions() -> list[str]:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    return sorted(set(re.findall(r'- "(3\.\d+)"', workflow)))


def test_python_version_support_matches_current_bindings() -> None:
    expected = ["3.10", "3.11", "3.12", "3.13"]

    assert pyproject_python_versions() == expected
    assert workflow_python_versions() == expected
