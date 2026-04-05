from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def pyproject_python_versions() -> list[str]:
    pyproject = (ROOT / "pyproject.toml").read_text()
    return sorted(
        set(
            re.findall(
                r'^\s*"Programming Language :: Python :: (3\.\d+)",$',
                pyproject,
                re.MULTILINE,
            )
        )
    )


def workflow_python_versions() -> list[str]:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    return sorted(set(re.findall(r'- "(3\.\d+)"', workflow)))


def workflow_contents() -> str:
    return (ROOT / ".github/workflows/ci.yml").read_text()


def test_python_version_support_matches_current_bindings() -> None:
    expected = ["3.10", "3.11", "3.12", "3.13"]

    assert pyproject_python_versions() == expected
    assert workflow_python_versions() == expected


def test_ci_avoids_maturin_develop_without_virtualenv() -> None:
    workflow = workflow_contents()

    assert "maturin develop" not in workflow
    assert "python -m pip install ." in workflow


def test_reference_module_is_vendored_in_repo() -> None:
    triangulation_test = (ROOT / "tests/test_triangulation.py").read_text()

    assert "/tmp/python_triangulation_reference.py" not in triangulation_test
    assert (ROOT / "tests/python_triangulation_reference.py").exists()
