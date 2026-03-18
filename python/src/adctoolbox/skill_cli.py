"""Install bundled Codex skills that ship with ADCToolbox."""

from __future__ import annotations

import argparse
import os
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


def default_skill_install_root() -> Path:
    """Return the default Codex skill installation directory."""
    codex_home = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    return codex_home / "skills"


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


def install_bundled_skills(
    names: list[str] | None = None,
    *,
    install_dev: bool = False,
    install_all: bool = False,
    dest: str | Path | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Install bundled skills into a Codex skills directory."""
    resolved_names = resolve_skill_names(
        names,
        install_dev=install_dev,
        install_all=install_all,
    )
    install_root = Path(dest).expanduser() if dest is not None else default_skill_install_root()
    install_root.mkdir(parents=True, exist_ok=True)

    installed_paths: list[Path] = []

    for skill_name in resolved_names:
        target_dir = install_root / skill_name
        skill_entry = resources.files("adctoolbox._bundled_skills").joinpath(
            "skills", skill_name, "SKILL.md"
        )

        if target_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Destination already exists: {target_dir}. "
                    "Use --force to replace it."
                )
            shutil.rmtree(target_dir)

        with resources.as_file(skill_entry) as skill_entry_path:
            shutil.copytree(skill_entry_path.parent, target_dir)
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
        help="Override the target Codex skills directory. Defaults to $CODEX_HOME/skills.",
    )
    parser.add_argument(
        "--force",
        "--upgrade",
        action="store_true",
        dest="force",
        help="Replace an existing installed skill directory.",
    )
    return parser


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

    try:
        installed_paths = install_bundled_skills(
            args.skills,
            install_dev=args.dev,
            install_all=args.all,
            dest=args.dest,
            overwrite=args.force,
        )
    except (FileExistsError, ValueError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1

    print("Installed ADCToolbox Codex skills:")
    for path in installed_paths:
        print(f"- {path}")
    print("Restart Codex to pick up new or updated skills.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
