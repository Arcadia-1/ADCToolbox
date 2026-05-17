"""Install bundled Codex skills that ship with ADCToolbox."""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from importlib import resources
from pathlib import Path


DEFAULT_SKILLS = ("adctoolbox-user-guide",)
DEV_SKILLS = ("adctoolbox-user-guide", "adctoolbox-contributor-guide")
AVAILABLE_SKILLS = ("adctoolbox-user-guide", "adctoolbox-contributor-guide")
SKILL_ALIASES = {
    "user": "adctoolbox-user-guide",
    "user-guide": "adctoolbox-user-guide",
    "adctoolbox-user-guide": "adctoolbox-user-guide",
    "dev": "adctoolbox-contributor-guide",
    "contributor": "adctoolbox-contributor-guide",
    "contributor-guide": "adctoolbox-contributor-guide",
    "adctoolbox-contributor-guide": "adctoolbox-contributor-guide",
}


def available_skill_names() -> list[str]:
    """Return bundled skill directory names in deterministic order."""
    return list(AVAILABLE_SKILLS)


def resolve_skill_names(
    names: list[str] | None = None,
    *,
    install_dev: bool = False,
    install_all: bool = False,
) -> list[str]:
    """Resolve requested skill names from aliases and mode flags."""
    if install_all:
        return available_skill_names()

    if install_dev:
        return list(DEV_SKILLS)

    if not names:
        return list(DEFAULT_SKILLS)

    resolved: list[str] = []
    available = set(available_skill_names())
    for raw_name in names:
        resolved_name = SKILL_ALIASES.get(raw_name, raw_name)
        if resolved_name not in available:
            available_display = ", ".join(sorted(available))
            raise ValueError(
                f"Unknown skill '{raw_name}'. Available skills: {available_display}"
            )
        if resolved_name not in resolved:
            resolved.append(resolved_name)
    return resolved


def bundled_skill_dir(skill_name: str) -> Path:
    """Return the filesystem path for a bundled skill directory."""
    skill_dir = resources.files("adctoolbox._bundled_skills").joinpath(
        "skills", skill_name
    )
    if not isinstance(skill_dir, Path):
        raise RuntimeError(
            "Bundled skill resources are not filesystem-backed; "
            "--editable symlink installation is unavailable."
        )
    return skill_dir


def _dirs_equal(left: Path, right: Path) -> bool:
    """Return True when two small directory trees have identical file content."""
    cmp = filecmp.dircmp(left, right)
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return False
    for name in cmp.common_files:
        if not filecmp.cmp(left / name, right / name, shallow=False):
            return False
    return all(_dirs_equal(left / name, right / name) for name in cmp.common_dirs)


def skill_status(
    names: list[str] | None = None,
    *,
    install_dev: bool = False,
    install_all: bool = False,
    dest: str | Path,
) -> list[dict[str, str | bool]]:
    """Inspect installed bundled skills under an explicit destination."""
    resolved_names = resolve_skill_names(
        names,
        install_dev=install_dev,
        install_all=install_all,
    )
    install_root = Path(dest).expanduser()
    rows: list[dict[str, str | bool]] = []

    for skill_name in resolved_names:
        source_dir = bundled_skill_dir(skill_name)
        target_dir = install_root / skill_name
        installed = target_dir.exists() or target_dir.is_symlink()
        mode = "missing"
        synced = False

        if target_dir.is_symlink():
            mode = "editable-link"
            try:
                synced = target_dir.resolve() == source_dir.resolve()
            except FileNotFoundError:
                synced = False
        elif target_dir.is_dir():
            mode = "copy"
            synced = _dirs_equal(source_dir, target_dir)
        elif target_dir.exists():
            mode = "non-directory"

        rows.append(
            {
                "name": skill_name,
                "source": str(source_dir),
                "target": str(target_dir),
                "installed": installed,
                "mode": mode,
                "synced": synced,
            }
        )

    return rows


def install_bundled_skills(
    names: list[str] | None = None,
    *,
    install_dev: bool = False,
    install_all: bool = False,
    dest: str | Path,
    overwrite: bool = False,
    editable: bool = False,
) -> list[Path]:
    """Install bundled skills into an explicit Codex skills directory."""
    resolved_names = resolve_skill_names(
        names,
        install_dev=install_dev,
        install_all=install_all,
    )
    install_root = Path(dest).expanduser()
    install_root.mkdir(parents=True, exist_ok=True)

    installed_paths: list[Path] = []

    for skill_name in resolved_names:
        target_dir = install_root / skill_name
        source_dir = bundled_skill_dir(skill_name)

        if target_dir.exists() or target_dir.is_symlink():
            if not overwrite:
                raise FileExistsError(
                    f"Destination already exists: {target_dir}. "
                    "Use --force to replace it."
                )
            if target_dir.is_symlink() or target_dir.is_file():
                target_dir.unlink()
            else:
                shutil.rmtree(target_dir)

        if editable:
            target_dir.symlink_to(source_dir, target_is_directory=True)
        else:
            shutil.copytree(source_dir, target_dir)
        installed_paths.append(target_dir)

    return installed_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install bundled ADCToolbox Codex skills.",
    )
    parser.add_argument(
        "skills",
        nargs="*",
        help=(
            "Skill names to install. Defaults to adctoolbox-user-guide. "
            "Use --dev to also install the maintainer skill."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List bundled skills and exit.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show install status for bundled skills under --dest and exit.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Install the default user skill plus the maintainer-only contributor skill.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Install all bundled skills.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        help="Required target Codex skills directory, e.g. ~/.codex/skills.",
    )
    parser.add_argument(
        "--force",
        "--upgrade",
        action="store_true",
        dest="force",
        help="Replace an existing installed skill directory.",
    )
    parser.add_argument(
        "--editable",
        action="store_true",
        help="Install skills as symlinks to the bundled source directories.",
    )
    return parser


def _require_dest(parser: argparse.ArgumentParser, dest: Path | None) -> Path:
    if dest is None:
        parser.error("--dest is required; this command never installs to a default path")
    return dest


def _print_status(rows: list[dict[str, str | bool]]) -> None:
    print("ADCToolbox Codex skill status:")
    for row in rows:
        installed = "yes" if row["installed"] else "no"
        synced = "yes" if row["synced"] else "no"
        print(f"- {row['name']}")
        print(f"  source: {row['source']}")
        print(f"  target: {row['target']}")
        print(f"  installed: {installed}")
        print(f"  mode: {row['mode']}")
        print(f"  synced: {synced}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print("Bundled ADCToolbox skills:")
        for name in available_skill_names():
            suffix = " (default)" if name in DEFAULT_SKILLS else ""
            if name == "adctoolbox-contributor-guide":
                suffix = " (dev-only)"
            print(f"- {name}{suffix}")
        return 0

    dest = _require_dest(parser, args.dest)

    if args.status:
        rows = skill_status(
            args.skills,
            install_dev=args.dev,
            install_all=args.all,
            dest=dest,
        )
        _print_status(rows)
        return 0

    try:
        installed_paths = install_bundled_skills(
            args.skills,
            install_dev=args.dev,
            install_all=args.all,
            dest=dest,
            overwrite=args.force,
            editable=args.editable,
        )
    except (FileExistsError, ValueError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1

    print("Installed ADCToolbox Codex skills:")
    for path in installed_paths:
        print(f"- {path}")
    if args.editable:
        print("Mode: editable symlink")
    else:
        print("Mode: copied directory")
    print("Restart Codex to pick up new or updated skills.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
