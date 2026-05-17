# ADCToolbox Agent Notes

## Codex Skills

Bundled Codex skills live in:

```text
python/src/adctoolbox/_bundled_skills/skills/
```

That bundled directory is the source of truth. Installed skills under a Codex
skills directory are generated copies or editable symlinks; do not edit the
installed copy as the authoritative source.

Available bundled skills:

- `adctoolbox-user-guide` — default user-facing skill
- `adctoolbox-contributor-guide` — maintainer-only skill, install only with
  `--dev` or `--all`

The installer never writes to a default path. Always pass an explicit
destination:

```bash
cd python
uv run adctoolbox-install-skill --dest ~/.codex/skills
```

Check what is installed and whether it matches the bundled source:

```bash
cd python
uv run adctoolbox-install-skill --status --dest ~/.codex/skills
```

For local skill development, use editable symlinks so source edits are reflected
without reinstalling:

```bash
cd python
uv run adctoolbox-install-skill --dev --editable --force --dest ~/.codex/skills
```

Use copied installs for normal users and editable installs only for local
development. Restart Codex after changing installed skills.

When changing bundled skills:

1. Edit files under `python/src/adctoolbox/_bundled_skills/skills/`.
2. Update smoke tests when examples or API usage snippets change.
3. Validate installer behavior with a temp destination, not the real
   `~/.codex/skills`.
4. Run:

```bash
cd python
uv run --with pytest pytest tests/unit/test_skill_cli.py -q
uv run --with pytest pytest tests/integration/test_user_guide_skill_examples.py -q
```
