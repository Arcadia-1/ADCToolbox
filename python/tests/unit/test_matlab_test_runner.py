import importlib.util
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace


def _make_executable(path: Path) -> Path:
    if os.name == "nt":
        path = path.with_suffix(".exe")
        shutil.copyfile(sys.executable, path)
        return path
    path.write_text("")
    path.chmod(0o755)
    return path


def _load_runner():
    script_path = Path(__file__).resolve().parents[3] / "matlab" / "tests" / "run_matlab_tests.py"
    spec = importlib.util.spec_from_file_location("matlab_test_runner", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_matlab_returns_skip_code(monkeypatch, capsys):
    runner = _load_runner()
    monkeypatch.delenv("ADCTOOLBOX_MATLAB", raising=False)
    monkeypatch.delenv("MATLAB_EXECUTABLE", raising=False)
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)
    monkeypatch.setattr(runner, "_standard_install_candidates", lambda: [])

    exit_code = runner.main(["common"])

    assert exit_code == runner.MISSING_MATLAB_EXIT_CODE
    assert "MATLAB executable not found" in capsys.readouterr().err


def test_missing_matlab_can_be_allowed(monkeypatch):
    runner = _load_runner()
    monkeypatch.delenv("ADCTOOLBOX_MATLAB", raising=False)
    monkeypatch.delenv("MATLAB_EXECUTABLE", raising=False)
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)
    monkeypatch.setattr(runner, "_standard_install_candidates", lambda: [])

    assert runner.main(["common", "--missing-ok"]) == 0


def test_explicit_missing_matlab_executable_does_not_fall_back(monkeypatch, tmp_path, capsys):
    runner = _load_runner()
    discovered_matlab = _make_executable(tmp_path / "matlab")
    monkeypatch.setattr(
        runner.shutil,
        "which",
        lambda name: str(discovered_matlab) if name in {"matlab", "matlab.exe"} else None,
    )
    monkeypatch.setattr(runner, "_standard_install_candidates", lambda: [discovered_matlab])

    exit_code = runner.main(["common", "--matlab-executable", str(tmp_path / "missing-matlab")])

    assert exit_code == runner.INVALID_MATLAB_EXECUTABLE_EXIT_CODE
    err = capsys.readouterr().err
    assert "--matlab-executable was not found" in err
    assert "automatic fallback discovery was not used" in err


def test_explicit_non_executable_matlab_path_is_rejected(tmp_path, capsys):
    runner = _load_runner()
    executable = tmp_path / "matlab"
    executable.write_text("")

    exit_code = runner.main(["common", "--matlab-executable", str(executable), "--dry-run"])

    assert exit_code == runner.INVALID_MATLAB_EXECUTABLE_EXIT_CODE
    assert "not found or is not executable" in capsys.readouterr().err


def test_explicit_invalid_windows_exe_is_rejected(monkeypatch, tmp_path, capsys):
    runner = _load_runner()
    monkeypatch.setattr(runner.os, "name", "nt")
    executable = tmp_path / "matlab.exe"
    executable.write_text("")

    exit_code = runner.main(["common", "--matlab-executable", str(executable), "--dry-run"])

    assert exit_code == runner.INVALID_MATLAB_EXECUTABLE_EXIT_CODE
    assert "not found or is not executable" in capsys.readouterr().err


def test_which_result_is_rechecked_before_accepting(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner.os, "name", "nt")
    executable = tmp_path / "matlab.exe"
    executable.write_text("")
    monkeypatch.setattr(runner.shutil, "which", lambda _name: str(executable))

    assert runner.find_matlab_executable("matlab") is None


def test_dry_run_prints_command_without_executing(tmp_path, capsys):
    runner = _load_runner()
    executable = _make_executable(tmp_path / "matlab")

    exit_code = runner.main(["aout", "--matlab-executable", str(executable), "--dry-run"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert str(executable) in output
    assert "run_aout" in output


def test_command_print_is_flushed_before_running(monkeypatch, tmp_path):
    runner = _load_runner()
    executable = _make_executable(tmp_path / "matlab")
    print_calls = []

    def fake_print(*args, **kwargs):
        print_calls.append((args, kwargs))

    monkeypatch.setattr("builtins.print", fake_print)
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))

    exit_code = runner.main(["common", "--matlab-executable", str(executable)])

    assert exit_code == 0
    assert print_calls[0][1]["flush"] is True
    assert "run_common" in print_calls[0][0][0]


def test_oserror_while_running_returns_invalid_executable(monkeypatch, tmp_path, capsys):
    runner = _load_runner()
    executable = _make_executable(tmp_path / "matlab")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot execute")),
    )

    exit_code = runner.main(["common", "--matlab-executable", str(executable)])

    assert exit_code == runner.INVALID_MATLAB_EXECUTABLE_EXIT_CODE
    assert "Failed to execute MATLAB executable" in capsys.readouterr().err


def test_build_command_escapes_matlab_root(tmp_path):
    runner = _load_runner()
    executable = tmp_path / "matlab"
    matlab_root = tmp_path / "ADCToolbox's matlab"

    command = runner.build_matlab_command(executable, "dout", matlab_root=matlab_root)

    assert command[0] == str(executable)
    assert command[1] == "-batch"
    assert "ADCToolbox''s matlab" in command[2]
    assert command[2].endswith("run_dout")
