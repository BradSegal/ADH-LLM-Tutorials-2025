"""Smart dependency installation for notebooks.

This module provides intelligent package installation that only installs
missing dependencies, reducing installation time and potential version conflicts
in Google Colab and local environments.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import tomllib

# Map distribution names to import names for packages with mismatches
PACKAGE_IMPORT_MAP = {
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
}


def is_package_installed(package_name: str) -> bool:
    """Check if a package is available for import.

    Uses importlib.util.find_spec() for fast, non-intrusive detection.
    This is approximately 1500x faster than actually importing the package.

    Args:
        package_name: Distribution name from pyproject.toml (e.g., 'scikit-learn')

    Returns:
        True if the package can be imported, False otherwise

    Examples:
        >>> is_package_installed("pandas")
        True
        >>> is_package_installed("nonexistent_package_xyz")
        False
        >>> is_package_installed("scikit-learn")  # Maps to 'sklearn'
        True
    """
    # Normalize package name for import check
    import_name = PACKAGE_IMPORT_MAP.get(package_name, package_name)

    try:
        spec = importlib.util.find_spec(import_name)
        return spec is not None
    except (ModuleNotFoundError, ImportError, ValueError):
        # ValueError can occur with malformed package names
        return False


def parse_dependencies(
    pyproject_path: Path,
    include_dev: bool = False,
) -> list[str]:
    """Extract dependency list from pyproject.toml.

    Args:
        pyproject_path: Path to pyproject.toml
        include_dev: Whether to include [dev] optional dependencies

    Returns:
        List of package names (may include version specifiers)

    Raises:
        FileNotFoundError: If pyproject.toml doesn't exist
        KeyError: If required fields are missing from pyproject.toml

    Examples:
        >>> parse_dependencies(Path("pyproject.toml"))
        ['torch', 'pandas', 'numpy', ...]
        >>> parse_dependencies(Path("pyproject.toml"), include_dev=True)
        ['torch', 'pandas', 'pytest', 'black', ...]
    """
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    # Get base dependencies
    deps = list(data["project"]["dependencies"])

    # Add dev dependencies if requested
    if include_dev:
        dev_deps = data["project"].get("optional-dependencies", {}).get("dev", [])
        deps.extend(dev_deps)

    return deps


def extract_package_name(dep_spec: str) -> str:
    """Extract base package name from a dependency specification.

    Handles various dependency specification formats including version
    constraints, extras, and URLs.

    Args:
        dep_spec: Dependency specification from pyproject.toml

    Returns:
        Base package name without version specifiers or extras

    Examples:
        >>> extract_package_name('torch')
        'torch'
        >>> extract_package_name('torch>=2.0')
        'torch'
        >>> extract_package_name('requests[security]>=2.28')
        'requests'
        >>> extract_package_name('package @ git+https://...')
        'package'
    """
    # Split on common version operators and extras markers
    for separator in [">=", "<=", "==", "!=", ">", "<", "~=", "[", "@"]:
        if separator in dep_spec:
            dep_spec = dep_spec.split(separator)[0]

    return dep_spec.strip()


def install_packages(
    packages: list[str],
    quiet: bool = True,
) -> tuple[list[str], list[str]]:
    """Install packages using pip.

    Args:
        packages: List of package names/specs to install
        quiet: Whether to suppress pip output

    Returns:
        Tuple of (successfully_installed, failed)

    Examples:
        >>> install_packages(['pandas', 'numpy'], quiet=True)
        (['pandas', 'numpy'], [])
    """
    if not packages:
        return [], []

    cmd = [sys.executable, "-m", "pip", "install"]

    if quiet:
        cmd.append("-q")

    cmd.extend(packages)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        return packages, []
    else:
        # If batch install fails, report all as failed
        # (could be enhanced to retry individually)
        return [], packages


def install_editable_package(
    repo_path: Path,
    include_dev: bool = False,
    quiet: bool = True,
) -> bool:
    """Install the local package in editable mode.

    Only installs if not already installed in editable mode from the
    specified repository path.

    Args:
        repo_path: Path to repository root (containing pyproject.toml)
        include_dev: Whether to install [dev] extras
        quiet: Whether to suppress pip output

    Returns:
        True if installation succeeded or already installed

    Examples:
        >>> install_editable_package(Path("/path/to/repo"), include_dev=True)
        True
    """
    # Check if already installed as editable from this location
    try:
        import importlib.metadata as metadata

        dist = metadata.distribution("digital-health-tutorial")

        # Check if it's pointing to our repo
        # Editable installs have a direct_url.json
        try:
            direct_url = dist.read_text("direct_url.json")
            if direct_url and str(repo_path) in direct_url:
                return True  # Already installed from this location
        except FileNotFoundError:
            # No direct_url.json, might be older install format
            pass
    except Exception:
        pass  # Not installed or can't determine, proceed with install

    # Install in editable mode
    cmd = [sys.executable, "-m", "pip", "install"]

    if quiet:
        cmd.append("-q")

    cmd.append("-e")

    if include_dev:
        cmd.append(f"{repo_path}[dev]")
    else:
        cmd.append(str(repo_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def fix_numpy_binary_compatibility(verbose: bool = True) -> bool:
    """Fix numpy binary compatibility issues in Google Colab.

    This addresses the common "numpy.dtype size changed" error that occurs
    when packages compiled against different numpy versions are loaded together.
    We force-upgrade numpy after all other packages are installed to ensure
    binary compatibility.

    Args:
        verbose: Whether to print progress messages

    Returns:
        True if upgrade succeeded, False otherwise
    """
    if verbose:
        print("\n🔧 Ensuring numpy binary compatibility...")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--no-cache-dir",
        "numpy",
    ]

    if verbose:
        print("   Running: pip install --upgrade --force-reinstall numpy")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        if verbose:
            print("✅ Numpy binary compatibility ensured")
        return True
    else:
        if verbose:
            print("⚠️  Warning: Could not upgrade numpy")
            print(f"   Error: {result.stderr[:200]}")
        return False


def smart_install_dependencies(
    repo_path: Path,
    include_dev: bool = False,
    verbose: bool = True,
    install_editable: bool = False,
) -> dict[str, list[str]]:
    """Intelligently install only missing dependencies.

    This is the main entry point for selective package installation.
    It checks each dependency individually and only installs packages
    that are not already available in the environment.

    Args:
        repo_path: Path to repository root
        include_dev: Whether to install dev dependencies
        verbose: Whether to print progress messages
        install_editable: Whether to install the package in editable mode
            (default: False)

    Returns:
        Dictionary with keys:
            - 'already_installed': Packages that were already present
            - 'newly_installed': Packages that were installed
            - 'failed': Packages that failed to install

    Raises:
        FileNotFoundError: If pyproject.toml not found at repo_path

    Examples:
        >>> result = smart_install_dependencies(
        ...     repo_path=Path("/path/to/repo"),
        ...     include_dev=True,
        ...     verbose=True
        ... )
        >>> result['already_installed']
        ['pandas', 'numpy', ...]
        >>> result['newly_installed']
        ['torch', ...]
        >>> result['failed']
        []
    """
    pyproject_path = repo_path / "pyproject.toml"

    if not pyproject_path.exists():
        raise FileNotFoundError(
            f"pyproject.toml not found at {pyproject_path}. "
            "Repository structure may be invalid."
        )

    # Parse dependencies
    dep_specs = parse_dependencies(pyproject_path, include_dev=include_dev)

    if verbose:
        print(f"📦 Checking {len(dep_specs)} dependencies...")

    # Separate into installed and missing
    already_installed = []
    missing = []

    for dep_spec in dep_specs:
        pkg_name = extract_package_name(dep_spec)

        if is_package_installed(pkg_name):
            already_installed.append(pkg_name)
        else:
            missing.append(dep_spec)  # Keep full spec for installation

    if verbose:
        print(f"✅ Already installed: {len(already_installed)}")
        print(f"📥 Need to install: {len(missing)}")

    # Install missing packages
    newly_installed: list[str] = []
    failed: list[str] = []

    if missing:
        if verbose:
            print(f"\n📦 Installing {len(missing)} missing packages...")
            for pkg in missing:
                print(f"  • {pkg}")

        # Batch install for speed (pip can parallelize)
        successful, failed = install_packages(missing, quiet=not verbose)
        newly_installed.extend(successful)

    # Handle editable install of local package (optional)
    editable_success = True
    if install_editable:
        if verbose:
            print("\n📦 Installing local 'core' package (editable mode)...")

        editable_success = install_editable_package(
            repo_path,
            include_dev=include_dev,
            quiet=not verbose,
        )

        if not editable_success:
            failed.append("digital-health-tutorial (editable)")

    # Fix numpy binary compatibility issues (common in Google Colab)
    # This prevents "numpy.dtype size changed" errors when importing packages
    # that have C extensions compiled against different numpy versions
    numpy_fix_success = fix_numpy_binary_compatibility(verbose=verbose)
    if not numpy_fix_success and verbose:
        print(
            "\n⚠️  Warning: Could not fix numpy compatibility. "
            "You may encounter import errors."
        )

    # Report results
    if verbose:
        print("\n" + "=" * 70)
        if not missing and editable_success:
            print("✅ All dependencies already satisfied!")
        else:
            print("✅ Installation complete!")
            print(f"   • Already present: {len(already_installed)}")
            print(f"   • Newly installed: {len(newly_installed)}")
            if failed:
                print(f"   ⚠️  Failed: {len(failed)}")
                for pkg in failed:
                    print(f"     - {pkg}")
        print("=" * 70)

    return {
        "already_installed": already_installed,
        "newly_installed": newly_installed,
        "failed": failed,
    }
