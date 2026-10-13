import ast
from pathlib import Path
import pytest
from spf_ssp126.ensemble import annual_index_mean

ROOT = Path(__file__).resolve().parents[1]


def test_python_syntax_and_code_only():
    paths = list((ROOT / "spf_ssp126").rglob("*.py")) + list((ROOT / "tests").rglob("*.py"))
    for path in paths:
        if path.name.startswith("._"):
            continue
        source = path.read_text()
        ast.parse(source)
    for path in (ROOT / "spf_ssp126").glob("*.py"):
        if path.name.startswith("._"):
            continue
        source = path.read_text()
        assert "/NHNHOME/" not in source
        assert "/Users/" not in source
        assert "import sfno" not in source.lower()
        assert "import diffesm" not in source.lower()
        assert "temporal_least_squares" not in source


def test_ensemble_rejects_partial_and_duplicate_members(tmp_path):
    with pytest.raises(ValueError):
        annual_index_mean([tmp_path / "one.npz"], expected_members=26)
    with pytest.raises(ValueError):
        annual_index_mean([tmp_path / "one.npz"] * 26)
