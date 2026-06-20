"""Run ADCToolbox MATLAB test suites with explicit environment checks."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


MISSING_MATLAB_EXIT_CODE = 77
INVALID_MATLAB_EXECUTABLE_EXIT_CODE = 2

SUITES = {
    "all": "run_all",
    "common": "run_common",
    "aout": "run_aout",
    "dout": "run_dout",
    "jitter": "run_jitter_load",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _matlab_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _matlab_quote(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _standard_install_candidates() -> list[Path]:
    candidates: list[Path] = []

    if os.name == "nt":
        for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
            root = os.environ.get(env_name)
            if not root:
                continue
            matlab_dir = Path(root) / "MATLAB"
            candidates.extend(sorted(matlab_dir.glob("R*/bin/matlab.exe"), reverse=True))
    elif sys.platform == "darwin":
        candidates.extend(sorted(Path("/Applications").glob("MATLAB_R*.app/bin/matlab"), reverse=True))
    else:
        candidates.extend(sorted(Path("/usr/local/MATLAB").glob("R*/bin/matlab"), reverse=True))

    return candidates


def _resolve_matlab_candidate(candidate: str) -> Path | None:
    candidate_path = Path(candidate)
    if candidate_path.exists():
        return candidate_path

    resolved = shutil.which(candidate)
    if resolved:
        return Path(resolved)

    return None


def find_matlab_executable(explicit: str | None = None) -> Path | None:
    """Find a MATLAB executable from an explicit path, env var, PATH, or standard installs."""
    if explicit:
        return _resolve_matlab_candidate(explicit)

    for env_name in ("ADCTOOLBOX_MATLAB", "MATLAB_EXECUTABLE"):
        env_value = os.environ.get(env_name)
        if env_value:
            resolved = _resolve_matlab_candidate(env_value)
            if resolved:
                return resolved

    for executable_name in ("matlab", "matlab.exe"):
        resolved = _resolve_matlab_candidate(executable_name)
        if resolved:
            return resolved

    for candidate_path in _standard_install_candidates():
        if candidate_path.exists():
            return candidate_path

    return None


def build_matlab_command(executable: Path, suite: str, matlab_root: Path | None = None) -> list[str]:
    suite_function = SUITES[suite]
    root = _matlab_root() if matlab_root is None else matlab_root
    matlab_statement = (
        f"cd('{_matlab_quote(root)}'); "
        "addpath(genpath(fullfile(pwd,'src'))); "
        "addpath(genpath(fullfile(pwd,'tests'))); "
        f"{suite_function}"
    )
    return [str(executable), "-batch", matlab_statement]


def _missing_matlab_message() -> str:
    return (
        "MATLAB executable not found. Install MATLAB R2020a or newer, add it to PATH, "
        "set ADCTOOLBOX_MATLAB/MATLAB_EXECUTABLE, or pass --matlab-executable. "
        f"Returning {MISSING_MATLAB_EXIT_CODE} to mark this optional external test "
        "environment as unavailable."
    )


def _invalid_matlab_executable_message(candidate: str) -> str:
    return (
        f"MATLAB executable specified by --matlab-executable was not found: {candidate}. "
        "Because the executable was set explicitly, automatic fallback discovery was not used."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "suite",
        choices=sorted(SUITES),
        nargs="?",
        default="all",
        help="MATLAB test suite to run.",
    )
    parser.add_argument(
        "--matlab-executable",
        help="Path to MATLAB executable. Overrides PATH discovery and environment variables.",
    )
    parser.add_argument(
        "--missing-ok",
        action="store_true",
        help="Return success instead of 77 when MATLAB is unavailable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the MATLAB command without executing it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    executable = find_matlab_executable(args.matlab_executable)
    if executable is None and args.matlab_executable:
        print(_invalid_matlab_executable_message(args.matlab_executable), file=sys.stderr)
        return INVALID_MATLAB_EXECUTABLE_EXIT_CODE

    if executable is None:
        print(_missing_matlab_message(), file=sys.stderr)
        return 0 if args.missing_ok else MISSING_MATLAB_EXIT_CODE

    command = build_matlab_command(executable, args.suite)
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=_repo_root(), check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
