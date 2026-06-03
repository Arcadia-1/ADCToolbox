from pathlib import Path

import pytest

from adctoolbox.skill_cli import available_skill_names
from adctoolbox.skill_cli import install_bundled_skills
from adctoolbox.skill_cli import main
from adctoolbox.skill_cli import skill_status


def _skip_if_directory_symlinks_unavailable(tmp_path: Path) -> None:
    source = tmp_path / "symlink-source"
    target = tmp_path / "symlink-target"
    source.mkdir()

    try:
        target.symlink_to(source, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")
    else:
        target.unlink()


def test_list_bundled_skills_contains_default_and_dev_skill():
    names = available_skill_names()

    assert "adctoolbox-user-guide" in names
    assert "adctoolbox-contributor-guide" in names


def test_install_default_skill_to_custom_destination(tmp_path: Path):
    install_root = tmp_path / "skills"

    installed_paths = install_bundled_skills(dest=install_root)

    assert installed_paths == [install_root / "adctoolbox-user-guide"]
    assert (install_root / "adctoolbox-user-guide" / "SKILL.md").exists()
    assert not (install_root / "adctoolbox-contributor-guide").exists()


def test_install_dev_skills_installs_both_skill_directories(tmp_path: Path):
    install_root = tmp_path / "skills"

    installed_paths = install_bundled_skills(install_dev=True, dest=install_root)

    assert install_root / "adctoolbox-user-guide" in installed_paths
    assert install_root / "adctoolbox-contributor-guide" in installed_paths
    assert (install_root / "adctoolbox-contributor-guide" / "references" / "testing-guide.md").exists()


def test_status_reports_missing_copy_and_synced(tmp_path: Path):
    install_root = tmp_path / "skills"

    before = skill_status(dest=install_root)
    assert before[0]["name"] == "adctoolbox-user-guide"
    assert before[0]["installed"] is False
    assert before[0]["mode"] == "missing"
    assert before[0]["synced"] is False

    install_bundled_skills(dest=install_root)
    after = skill_status(dest=install_root)
    assert after[0]["installed"] is True
    assert after[0]["mode"] == "copy"
    assert after[0]["synced"] is True


def test_install_editable_skill_creates_symlink(tmp_path: Path):
    _skip_if_directory_symlinks_unavailable(tmp_path)

    install_root = tmp_path / "skills"

    installed_paths = install_bundled_skills(dest=install_root, editable=True)
    target = install_root / "adctoolbox-user-guide"

    assert installed_paths == [target]
    assert target.is_symlink()
    assert (target / "SKILL.md").exists()

    status = skill_status(dest=install_root)
    assert status[0]["mode"] == "editable-link"
    assert status[0]["synced"] is True


def test_cli_requires_dest_for_install(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])

    assert exc.value.code == 2
    assert "--dest is required" in capsys.readouterr().err


def test_cli_status_uses_explicit_dest(tmp_path: Path, capsys):
    result = main(["--status", "--dest", str(tmp_path / "skills")])

    assert result == 0
    out = capsys.readouterr().out
    assert "ADCToolbox Codex skill status:" in out
    assert "installed: no" in out
