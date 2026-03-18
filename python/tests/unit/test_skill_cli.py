from pathlib import Path

from adctoolbox.skill_cli import available_skill_names
from adctoolbox.skill_cli import install_bundled_skills


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
