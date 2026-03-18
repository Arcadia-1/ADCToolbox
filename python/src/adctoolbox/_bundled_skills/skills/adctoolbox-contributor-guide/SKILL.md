---
name: adctoolbox-contributor-guide
description: >
  Maintainer-only guide for editing the ADCToolbox repository itself. Use this
  skill only when changing source, exports, examples, tests, packaging, or the
  bundled skills under `python/src/adctoolbox/_bundled_skills/skills/`. Do not
  use it for ordinary library usage questions; use `adctoolbox-user-guide`
  for that.
---

# ADCToolbox Maintainer Guide

This skill is for repository maintenance only.

Use `adctoolbox-user-guide` when the task is:

- choosing a tool from the library
- writing user-facing Python example code
- analyzing ADC data with the package

Use this skill when the task is:

- adding or changing a public API
- modifying package exports
- adding tests or fixing broken tests
- changing packaging, CLI commands, or bundled skill assets
- updating repository examples or maintainer docs

For test execution details, open `references/testing-guide.md`.

## 1. Public API Maintenance

The top-level Python API is controlled by `python/src/adctoolbox/__init__.py`.

If you add a new public flat import:

1. add it in the relevant submodule
2. export it in the submodule `__init__.py`
3. register it in top-level `adctoolbox/__init__.py` with `_export(...)`
4. update examples and user-facing skill docs if the new API is user-visible

Do not expose helpers with leading `_`.

Some public tools are intentionally not flat-exported, such as:

- `adctoolbox.siggen.ADC_Signal_Generator`
- `adctoolbox.toolset.generate_aout_dashboard`
- `adctoolbox.toolset.generate_dout_dashboard`
- `adctoolbox.calibration.calibrate_weight_sine_lite`
- `adctoolbox.fundamentals.validate_aout_data`
- `adctoolbox.fundamentals.validate_dout_data`

If you change these interfaces, update the user guide skill because agents
rely on that routing.

## 2. Return-Shape Policy

For new Python public APIs, prefer `dict` returns with stable snake_case keys.

Do not introduce new tuple-style public returns unless there is a very strong
compatibility reason.

Current legacy exceptions still in the package include:

- `find_coherent_frequency`
- `analyze_bit_activity`
- `analyze_overflow`
- `analyze_enob_sweep`
- `fit_static_nonlin`
- `calibrate_weight_sine_lite`
- `convert_cap_to_weight`

Important correction:

- `analyze_weight_radix` currently returns a `dict`, not a bare array

If you change a return shape, update:

1. code
2. tests
3. examples
4. bundled skill docs under `python/src/adctoolbox/_bundled_skills/skills/`

## 3. Bundled Skills

Skills are maintained only under:

- `python/src/adctoolbox/_bundled_skills/skills/`

When editing a skill:

1. keep `SKILL.md` short and use references only when needed
2. prefer routing to packaged examples over duplicating long tutorials
3. verify the installer CLI still installs the expected files

## 4. What Must Stay In Sync

When a public API changes, do not stop at code changes.

- update affected examples under `python/src/adctoolbox/examples/`
- update bundled skill docs if the routing or return shape changed
- update tests that pin exports or return keys
- correct README snippets only when they would mislead users

Treat source exports and runnable examples as the canonical user-facing API.

## 5. Packaging And CLI Changes

If you add or modify a CLI:

- register it in `python/pyproject.toml`
- keep behavior non-destructive by default
- support explicit overwrite/upgrade flags for replacement flows
- validate with a temp destination, not the real `~/.codex/skills`

For Codex skill installation specifically:

- default install target should be `$CODEX_HOME/skills` or `~/.codex/skills`
- default install should only install `adctoolbox-user-guide`
- dev-only skill installation should require explicit `--dev` or `--all`
