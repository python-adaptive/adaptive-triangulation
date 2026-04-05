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


def pyproject_contents() -> str:
    return (ROOT / "pyproject.toml").read_text()


def test_python_version_support_matches_current_bindings() -> None:
    expected = ["3.10", "3.11", "3.12", "3.13"]

    assert pyproject_python_versions() == expected
    assert workflow_python_versions() == expected


def test_ci_uses_uv_sync_and_runs_pre_commit() -> None:
    workflow = workflow_contents()

    assert "maturin develop" not in workflow
    assert "python -m pip install ." not in workflow
    assert "astral-sh/setup-uv" in workflow
    assert "uv sync --locked --only-group test" in workflow
    assert "uv run pytest -x -v" in workflow
    assert "uv run pre-commit run --all-files" in workflow


def test_pyproject_declares_uv_dependency_groups() -> None:
    pyproject = pyproject_contents()

    assert "[dependency-groups]" in pyproject
    assert "test = [" in pyproject
    assert "lint = [" in pyproject
    assert '{ include-group = "lint" }' in pyproject
    assert '{ include-group = "test" }' in pyproject
    assert '"adaptive"' in pyproject
    assert '"pre-commit"' in pyproject


def test_reference_module_comes_from_installed_adaptive() -> None:
    triangulation_test = (ROOT / "tests/test_triangulation.py").read_text()

    assert "from adaptive.learner import triangulation as reference_module" in triangulation_test
    assert "python_triangulation_reference.py" not in triangulation_test
    assert not (ROOT / "tests/python_triangulation_reference.py").exists()
