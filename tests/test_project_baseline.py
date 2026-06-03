from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_open_source_baseline_files_exist():
    expected_paths = [
        "LICENSE",
        "CONTRIBUTING.md",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        ".github/workflows/ci.yml",
    ]

    missing = [path for path in expected_paths if not (ROOT / path).exists()]

    assert missing == []


def test_houseprice_uses_project_mini_d2l_instead_of_external_d2l():
    source = read_text("houseprice.py")

    assert "from d2l import" not in source
    assert "import mini_d2l as d2l" in source


def test_houseprice_keeps_training_behind_main_guard():
    source = read_text("houseprice.py")

    assert "def main(" in source
    assert 'if __name__ == "__main__":' in source


def test_chapter5_demo_gpu_uses_mlp_compatible_input_shape():
    source = read_text("chapter5.py")
    demo_gpu = re.search(r"def demo_gpu\(\):(?P<body>.*?)(?=\n\ndef |\n\nif __name__)", source, re.S)

    assert demo_gpu is not None
    assert "torch.ones((2, 20)" in demo_gpu.group("body")


def test_runtime_and_development_requirements_are_separated():
    runtime_requirements = read_text("requirements.txt").splitlines()
    dev_requirements = read_text("requirements-dev.txt").splitlines()

    for requirement in ["torch", "torchvision", "matplotlib", "numpy", "pandas"]:
        assert requirement in runtime_requirements

    assert "-r requirements.txt" in dev_requirements
    assert "pytest" in dev_requirements
    assert "ruff" in dev_requirements


def test_pyproject_declares_python_baseline_and_tooling():
    pyproject = read_text("pyproject.toml")

    assert 'requires-python = ">=3.10,<3.13"' in pyproject
    assert "[tool.pytest.ini_options]" in pyproject
    assert "[tool.ruff]" in pyproject


def test_ci_runs_compile_lint_and_pytest_on_supported_python_versions():
    workflow = read_text(".github/workflows/ci.yml")

    for version in ["3.10", "3.11", "3.12"]:
        assert version in workflow

    assert "python -m compileall -q ." in workflow
    assert "ruff check ." in workflow
    assert "pytest -q" in workflow
