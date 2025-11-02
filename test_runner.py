#!/usr/bin/env python3
"""
Test runner script for CrewAI A2A Adapter
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a command and return the result"""
    print(f"\n🔄 {description}...")
    print(f"Running: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )

        if result.returncode == 0:
            print(f"✅ {description} passed")
            if result.stdout.strip():
                print(f"Output:\n{result.stdout}")
            return True
        else:
            print(f"❌ {description} failed")
            if result.stdout.strip():
                print(f"STDOUT:\n{result.stdout}")
            if result.stderr.strip():
                print(f"STDERR:\n{result.stderr}")
            return False

    except FileNotFoundError as e:
        print(f"❌ Command not found: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def install_dependencies():
    """Install test dependencies"""
    print("📦 Installing dependencies...")

    # Install main dependencies
    deps = [
        "pytest>=8.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "httpx>=0.28.0",
        "pydantic>=2.0.0"
    ]

    for dep in deps:
        cmd = [sys.executable, "-m", "pip", "install", dep]
        if not run_command(cmd, f"Installing {dep}"):
            return False

    return True


def run_tests(test_path=None, verbose=False, coverage=False, marker=None):
    """Run the test suite"""
    cmd = [sys.executable, "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])

    if marker:
        cmd.extend(["-m", marker])

    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")

    return run_command(cmd, f"Running tests{f' with marker {marker}' if marker else ''}")


def run_linting():
    """Run code linting"""
    success = True

    # Try to run black
    try:
        if not run_command([sys.executable, "-m", "black", "--check", "src/"], "Code formatting check (black)"):
            print("💡 Run 'python -m black src/' to fix formatting")
            success = False
    except Exception:
        print("⚠️  Black not available, skipping code formatting check")

    # Try to run basic import tests
    if not run_command([sys.executable, "-c", "from src import *; print('All imports successful')"], "Import validation"):
        success = False

    return success


def run_type_checking():
    """Run type checking"""
    try:
        return run_command([sys.executable, "-m", "mypy", "src/"], "Type checking (mypy)")
    except Exception:
        print("⚠️  MyPy not available, skipping type checking")
        return True


def run_security_check():
    """Run basic security checks"""
    try:
        # Check for common security issues in dependencies
        return run_command([sys.executable, "-m", "pip", "check"], "Dependency security check")
    except Exception:
        print("⚠️  Could not run security check")
        return True


def validate_examples():
    """Validate that example files can be imported"""
    examples_dir = Path("examples")
    if not examples_dir.exists():
        print("⚠️  No examples directory found")
        return True

    success = True
    for example_file in examples_dir.glob("*.py"):
        # Basic syntax validation
        cmd = [sys.executable, "-m", "py_compile", str(example_file)]
        if not run_command(cmd, f"Validating {example_file.name}"):
            success = False

    return success


def main():
    parser = argparse.ArgumentParser(description="Test runner for CrewAI A2A Adapter")
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies")
    parser.add_argument("--tests-only", action="store_true", help="Run only tests")
    parser.add_argument("--lint-only", action="store_true", help="Run only linting")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--marker", "-m", help="Run tests with specific marker")
    parser.add_argument("--test-path", help="Run specific test file or directory")
    parser.add_argument("--quick", action="store_true", help="Quick test run (skip type checking and examples)")

    args = parser.parse_args()

    print("🚀 CrewAI A2A Adapter Test Suite")
    print("=" * 50)

    success = True

    # Install dependencies if requested
    if args.install_deps:
        if not install_dependencies():
            success = False

    # Run tests
    if not args.lint_only:
        if not run_tests(args.test_path, args.verbose, args.coverage, args.marker):
            success = False

    # Run linting
    if not args.tests_only:
        if not run_linting():
            success = False

    # Run additional checks (unless quick mode)
    if not args.quick and not args.tests_only:
        if not run_type_checking():
            success = False

        if not run_security_check():
            success = False

        if not validate_examples():
            success = False

    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 All checks passed!")
        sys.exit(0)
    else:
        print("💥 Some checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()