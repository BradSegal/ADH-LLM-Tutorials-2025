"""Unit tests for core.notebook.setup module."""

from pathlib import Path

import pytest

from core.notebook.setup import (
    extract_package_name,
    is_package_installed,
    parse_dependencies,
)


class TestIsPackageInstalled:
    """Tests for is_package_installed function."""

    def test_stdlib_package_is_installed(self) -> None:
        """Standard library packages should always be detected."""
        assert is_package_installed("sys") is True
        assert is_package_installed("os") is True
        assert is_package_installed("pathlib") is True

    def test_common_package_is_installed(self) -> None:
        """Common packages that should be installed should be detected."""
        # These are in our dependencies
        assert is_package_installed("pandas") is True
        assert is_package_installed("torch") is True

    def test_nonexistent_package_not_installed(self) -> None:
        """Nonexistent packages should not be detected."""
        assert is_package_installed("nonexistent_pkg_xyz_12345") is False
        assert is_package_installed("fake_package_name") is False

    def test_package_name_normalization_scikit_learn(self) -> None:
        """Test that scikit-learn is properly normalized to sklearn."""
        # scikit-learn is installed as 'scikit-learn' but imported as 'sklearn'
        assert is_package_installed("scikit-learn") is True

    def test_package_name_normalization_pyyaml(self) -> None:
        """Test that pyyaml is properly normalized to yaml."""
        # pyyaml is installed as 'pyyaml' but imported as 'yaml'
        assert is_package_installed("pyyaml") is True

    def test_malformed_package_name_returns_false(self) -> None:
        """Malformed package names should return False, not raise."""
        # Should handle gracefully without raising exceptions
        assert is_package_installed("") is False
        assert is_package_installed("...") is False


class TestExtractPackageName:
    """Tests for extract_package_name function."""

    def test_simple_package_name(self) -> None:
        """Simple package names should be returned as-is."""
        assert extract_package_name("torch") == "torch"
        assert extract_package_name("pandas") == "pandas"
        assert extract_package_name("numpy") == "numpy"

    def test_package_with_version_constraint_gte(self) -> None:
        """Package names with >= constraints should be extracted."""
        assert extract_package_name("torch>=2.0") == "torch"
        assert extract_package_name("pandas>=2.1.0") == "pandas"

    def test_package_with_version_constraint_lte(self) -> None:
        """Package names with <= constraints should be extracted."""
        assert extract_package_name("numpy<=1.24") == "numpy"

    def test_package_with_version_constraint_eq(self) -> None:
        """Package names with == constraints should be extracted."""
        assert extract_package_name("requests==2.28.0") == "requests"

    def test_package_with_extras(self) -> None:
        """Package names with extras should be extracted."""
        assert extract_package_name("requests[security]") == "requests"
        assert extract_package_name("requests[security]>=2.28") == "requests"

    def test_package_with_git_url(self) -> None:
        """Package names with git URLs should be extracted."""
        assert extract_package_name("package @ git+https://github.com/...") == "package"

    def test_package_with_multiple_constraints(self) -> None:
        """Only the first constraint separator should matter."""
        assert extract_package_name("package>=1.0,<2.0") == "package"

    def test_package_with_whitespace(self) -> None:
        """Whitespace should be stripped from package names."""
        assert extract_package_name("  torch  ") == "torch"
        assert extract_package_name("pandas >=2.0") == "pandas"


class TestParseDependencies:
    """Tests for parse_dependencies function."""

    def test_parse_base_dependencies_only(self, tmp_path: Path) -> None:
        """Should parse base dependencies from pyproject.toml."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[project]
dependencies = ["torch", "pandas>=2.0"]

[project.optional-dependencies]
dev = ["pytest", "black"]
"""
        )

        deps = parse_dependencies(pyproject, include_dev=False)
        assert len(deps) == 2
        assert "torch" in deps
        assert "pandas>=2.0" in deps
        assert "pytest" not in deps
        assert "black" not in deps

    def test_parse_with_dev_dependencies(self, tmp_path: Path) -> None:
        """Should parse base + dev dependencies when include_dev=True."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[project]
dependencies = ["torch", "pandas>=2.0"]

[project.optional-dependencies]
dev = ["pytest", "black"]
"""
        )

        deps = parse_dependencies(pyproject, include_dev=True)
        assert len(deps) == 4
        assert "torch" in deps
        assert "pandas>=2.0" in deps
        assert "pytest" in deps
        assert "black" in deps

    def test_parse_with_no_dev_section(self, tmp_path: Path) -> None:
        """Should handle pyproject.toml without [project.optional-dependencies]."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[project]
dependencies = ["torch", "pandas"]
"""
        )

        deps = parse_dependencies(pyproject, include_dev=True)
        assert len(deps) == 2
        assert "torch" in deps
        assert "pandas" in deps

    def test_parse_file_not_found(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for missing pyproject.toml."""
        nonexistent = tmp_path / "nonexistent.toml"

        with pytest.raises(FileNotFoundError):
            parse_dependencies(nonexistent)

    def test_parse_empty_dependencies(self, tmp_path: Path) -> None:
        """Should handle empty dependencies list."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[project]
dependencies = []
"""
        )

        deps = parse_dependencies(pyproject, include_dev=False)
        assert len(deps) == 0
        assert deps == []


class TestParseDependenciesWithRealFile:
    """Integration tests using the actual project's pyproject.toml."""

    def test_parse_real_pyproject_base(self) -> None:
        """Should parse the actual project's base dependencies."""
        repo_root = Path(__file__).parents[3]  # Go up to repo root
        pyproject = repo_root / "pyproject.toml"

        # Ensure the file exists
        assert pyproject.exists(), "pyproject.toml not found in repo root"

        deps = parse_dependencies(pyproject, include_dev=False)

        # Should have at least some expected base dependencies
        assert len(deps) > 0
        # Check for a few key dependencies we know should be there
        dep_names = [extract_package_name(d) for d in deps]
        assert "torch" in dep_names
        assert "pandas" in dep_names

    def test_parse_real_pyproject_with_dev(self) -> None:
        """Should parse the actual project's dependencies including dev."""
        repo_root = Path(__file__).parents[3]
        pyproject = repo_root / "pyproject.toml"

        deps = parse_dependencies(pyproject, include_dev=True)

        # Should have more dependencies with dev included
        dep_names = [extract_package_name(d) for d in deps]

        # Check for dev dependencies
        assert "pytest" in dep_names
        assert "black" in dep_names
        assert "ruff" in dep_names
        assert "mypy" in dep_names
