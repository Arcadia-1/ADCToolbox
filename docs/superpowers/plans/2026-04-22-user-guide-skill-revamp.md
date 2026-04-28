# `adctoolbox-user-guide` Skill Revamp — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the bundled `adctoolbox-user-guide` skill so its `SKILL.md` is a short, high-precision *basic tier* (spectrum + calibration + critical conventions) and advanced debug content lives in `references/advanced-debug.md`. Drive every edit with a measured `skill-creator` eval (10 fixed prompts + locked pass/fail criteria).

**Architecture:** Progressive Disclosure — Claude always has the basic tier in context; it opens the advanced reference only when a task needs dashboards, phase-plane / error-decomposition, bit-level analysis, static nonlinearity, or cap-to-weight conversion. Each implementation commit is gated by the eval and independently revertable.

**Tech Stack:** Markdown (skill content) + `skill-creator` plugin skill (benchmark / variance analysis) + git (`feature/user-guide-skill-revamp` branch, already created off `origin/main`).

**Spec:** `docs/superpowers/specs/2026-04-22-user-guide-skill-revamp-design.md` (commit `54221e4`).

---

## Prerequisites (verify before starting Task 1)

- [ ] **P1:** Confirm current branch is `feature/user-guide-skill-revamp` in `ADCToolbox/`.
  ```bash
  git -C ADCToolbox rev-parse --abbrev-ref HEAD
  # Expected: feature/user-guide-skill-revamp
  ```

- [ ] **P2:** Confirm working tree is clean (spec already committed).
  ```bash
  git -C ADCToolbox status --short
  # Expected: (empty output)
  ```

- [ ] **P3:** Confirm `skill-creator` is available via the Skill tool.
  It should appear in the skills list the harness injects at session start (look for `skill-creator:skill-creator`). If not, stop and report — eval cannot proceed.

- [ ] **P4:** Open the spec in a second pane; keep it reachable while working.
  `ADCToolbox/docs/superpowers/specs/2026-04-22-user-guide-skill-revamp-design.md`

---

## File Map (locked before any edits)

| Path (under `ADCToolbox/`) | Action | Responsibility |
|---|---|---|
| `docs/superpowers/specs/eval/prompts.md` | **Create** | 10 fixed eval prompts + success criteria + where to log per-iteration results |
| `docs/superpowers/specs/eval/baseline.json` | **Create** | skill-creator baseline output (before any SKILL.md edits) |
| `docs/superpowers/specs/eval/iter-N.json` | **Create** (per iteration) | skill-creator output after each candidate rewrite |
| `python/src/adctoolbox/_bundled_skills/skills/adctoolbox-user-guide/references/advanced-debug.md` | **Create** | Advanced debug tier: dashboards, phase-plane, bit-level, error decomposition, cap-to-weight |
| `python/src/adctoolbox/_bundled_skills/skills/adctoolbox-user-guide/SKILL.md` | **Rewrite** | Basic tier (spectrum + calibration + conventions). Drops "Installing AI Skills" section. Adds explicit "NOT for" disambiguation and escalation hook to advanced-debug.md |
| `python/src/adctoolbox/_bundled_skills/skills/adctoolbox-user-guide/references/api-quickref.md` | **Modify** | Re-partition into `## Basic` / `## Advanced` sections (no content removed) |
| `python/src/adctoolbox/_bundled_skills/skills/adctoolbox-user-guide/references/example-map.md` | **Modify** | Add `## Advanced examples` section; keep basic section as-is |

Shorthand used below: `UG = python/src/adctoolbox/_bundled_skills/skills/adctoolbox-user-guide`.

---

## Task 1: Lock the eval prompt suite

**Files:**
- Create: `docs/superpowers/specs/eval/prompts.md`

- [ ] **Step 1.1: Create the eval prompts file**

Write exactly this to `docs/superpowers/specs/eval/prompts.md`:

````markdown
# adctoolbox-user-guide — Eval Prompt Suite

Locked before baseline. Do NOT change these prompts after `baseline.json`
is captured — any change invalidates the comparison.

## Basic (5)

**B1.** Help me compute SNDR from this ADC dout array.
**B2.** Plot a spectrum of the ADC output with coherent sampling.
**B3.** I have a raw dout buffer — give me Python code that gets me ENOB.
**B4.** Calibrate SAR capacitor weights from a sine test.
**B5.** Generate a synthetic ADC output with a sinusoidal stimulus for testing.

## Advanced (5)

**A1.** Plot a phase-plane to diagnose nonlinearity in the ADC aout.
**A2.** Show me per-bit activity for an N-bit ADC code stream.
**A3.** How do I detect overflow events in dout?
**A4.** Give me a one-shot dashboard of all aout diagnostics.
**A5.** Convert a binary-weighted cap array to normalized weights.

## Success criteria (locked)

- **Basic:** ≥ 4/5 prompts where (a) the skill triggers, and (b) the
  response references the correct top-level API name (e.g. B1 mentions
  `analyze_spectrum` and its SNDR return key).
- **Advanced:** ≥ 3/5 prompts where the response explicitly escalates
  to `references/advanced-debug.md` (points to it or paraphrases
  opening it).
- **Average response length:** non-increasing vs baseline.

## Per-iteration log

| Iteration | Commit | Basic pass | Advanced pass | Avg length (words) | Artifact |
|-----------|--------|------------|----------------|--------------------|----------|
| baseline  | (pre-change) | TBD / 5 | TBD / 5 | TBD | `baseline.json` |
````

- [ ] **Step 1.2: Commit**

```bash
git -C ADCToolbox add docs/superpowers/specs/eval/prompts.md
git -C ADCToolbox commit -m "eval: lock prompt suite for user-guide skill revamp"
```

---

## Task 2: Capture baseline eval

**Files:**
- Create: `docs/superpowers/specs/eval/baseline.json`
- Modify: `docs/superpowers/specs/eval/prompts.md` (fill in the baseline row of the per-iteration log)

- [ ] **Step 2.1: Verify the baseline target file is the pre-revamp `SKILL.md`**

```bash
git -C ADCToolbox diff origin/main -- python/src/adctoolbox/_bundled_skills/skills/adctoolbox-user-guide/SKILL.md
# Expected: no diff (SKILL.md still matches origin/main)
```

If there is a diff, abort — the branch has already been edited and the baseline would not be a true baseline.

- [ ] **Step 2.2: Invoke `skill-creator` to run the baseline benchmark**

Use the Skill tool to invoke `skill-creator:skill-creator`. In the invocation, instruct it to:
1. Run its benchmark / performance-measurement mode
2. Target skill: `adctoolbox-user-guide` at path `ADCToolbox/python/src/adctoolbox/_bundled_skills/skills/adctoolbox-user-guide/`
3. Use the 10 prompts in `ADCToolbox/docs/superpowers/specs/eval/prompts.md` (sections "Basic" and "Advanced")
4. Output JSON results to `ADCToolbox/docs/superpowers/specs/eval/baseline.json` including, per prompt: `{prompt_id, triggered: bool, referenced_apis: [str], escalated_to_advanced: bool, response_length_words: int}`

If `skill-creator` does not support a direct "point at this SKILL.md + run these prompts" invocation and requires a different shape (e.g. it expects a staged working copy), follow its interactive prompts — but do **not** change the 10 prompts or the target SKILL.md.

- [ ] **Step 2.3: Fill in the baseline row in `prompts.md`**

Edit the "Per-iteration log" table in `docs/superpowers/specs/eval/prompts.md`:

```markdown
| baseline  | 687dbf1 (pre-change) | X / 5 | Y / 5 | Z | `baseline.json` |
```

Replace `X`, `Y`, `Z` with the counts and average word length computed from `baseline.json`. (If `skill-creator` reports a different commit hash for the baseline, use what it reports.)

- [ ] **Step 2.4: Commit**

```bash
git -C ADCToolbox add docs/superpowers/specs/eval/baseline.json docs/superpowers/specs/eval/prompts.md
git -C ADCToolbox commit -m "eval: capture baseline for user-guide skill"
```

---

## Task 3: Create `references/advanced-debug.md`

**Files:**
- Create: `python/src/adctoolbox/_bundled_skills/skills/adctoolbox-user-guide/references/advanced-debug.md`

- [ ] **Step 3.1: Enumerate the advanced API surface**

Read the public exports from these files (do **not** edit them):
- `ADCToolbox/python/src/adctoolbox/__init__.py`
- `ADCToolbox/python/src/adctoolbox/aout/__init__.py`
- `ADCToolbox/python/src/adctoolbox/toolset/__init__.py`
- `ADCToolbox/python/src/adctoolbox/dout/__init__.py`
- `ADCToolbox/python/src/adctoolbox/fundamentals/__init__.py`

Record (as a scratch note inside your working context, **not** a committed file) the concrete public names for each advanced category in the spec §2 partition. Known names to verify against the actual `__init__.py`:

- Dashboards: `generate_aout_dashboard`, `generate_dout_dashboard`
- Phase-plane: `analyze_phase_plane`, `analyze_error_phase_plane`
- Bit-level: `analyze_bit_activity`, `analyze_overflow`, `analyze_enob_sweep`, `analyze_weight_radix`
- Static nonlinearity: `fit_static_nonlin`
- Error decomposition (from `aout/`): `analyze_decomposition_polar`, `analyze_decomposition_time`, `analyze_error_autocorr`, `analyze_error_by_phase`, `analyze_error_by_value`, `analyze_error_envelope_spectrum`, `analyze_error_pdf`, `analyze_error_spectrum`, `analyze_inl_from_sine`, `decompose_harmonic_error`
- Cap-to-weight: `convert_cap_to_weight`

If a name in that list is **not** actually public (not re-exported in an `__init__.py`), drop it from the reference file — do not ship API names the user can't import.

- [ ] **Step 3.2: Write `advanced-debug.md`**

Create the file at `UG/references/advanced-debug.md` with the task-keyword-organized layout below. For each section, include import path, signature-style one-liner, return-shape note (only if non-uniform), and a ≤ 8-line snippet.

````markdown
# ADCToolbox — Advanced Debug Reference

Load this file when the basic spectrum/calibration tier in `SKILL.md`
is not enough. Each section below is keyed by the user's likely
question, not by file layout.

Import conventions follow `SKILL.md` §5. Frequency conventions follow
`SKILL.md` §2.

## "I want one image showing all aout/dout diagnostics"

Goal: generate a multi-plot dashboard (time-domain + spectrum + INL/DNL + extras).

```python
from adctoolbox.toolset import generate_aout_dashboard, generate_dout_dashboard

generate_aout_dashboard(aout, fs=fs, savepath="aout_dash.png")
generate_dout_dashboard(dout, n_bits=N, fs=fs, savepath="dout_dash.png")
```

Use `_aout_` when you have reconstructed analog output (floats); use
`_dout_` when you only have raw digital codes.

## "I need to see nonlinearity structure, not just a single INL/DNL number"

```python
from adctoolbox.aout import analyze_phase_plane, analyze_error_phase_plane

analyze_phase_plane(aout, fs=fs, Fin=Fin)           # full signal phase trajectory
analyze_error_phase_plane(aout, fs=fs, Fin=Fin)     # error-only phase plane
```

Use `analyze_error_phase_plane` after `fit_sine_4param` to isolate
nonlinearity from the fundamental.

## "I want per-bit behavior — activity, overflow, or ENOB vs bit depth"

```python
from adctoolbox import analyze_bit_activity, analyze_overflow, analyze_enob_sweep, analyze_weight_radix
```

- `analyze_bit_activity(dout, n_bits=N)` → `ndarray`
- `analyze_overflow(dout, n_bits=N)` → `tuple`
- `analyze_enob_sweep(dout, n_bits=N)` → `tuple (enob_sweep, n_bits_vec)`
- `analyze_weight_radix(weights)` → `dict`   (was a bare array in old versions — now a dict)

## "I have static INL/DNL data and want a nonlinearity fit"

```python
from adctoolbox import fit_static_nonlin
coef, residual = fit_static_nonlin(inl_or_dnl, order=3)   # returns tuple
```

## "I want to decompose total error into component contributions"

```python
from adctoolbox.aout import (
    analyze_decomposition_polar,
    analyze_decomposition_time,
    decompose_harmonic_error,
    analyze_error_spectrum,
    analyze_error_envelope_spectrum,
    analyze_error_pdf,
    analyze_error_autocorr,
    analyze_error_by_phase,
    analyze_error_by_value,
    analyze_inl_from_sine,
)
```

Pick by the error view you need:

- by phase / by value → `analyze_error_by_phase`, `analyze_error_by_value`
- harmonic decomposition → `decompose_harmonic_error`
- spectral view of the error → `analyze_error_spectrum`, `analyze_error_envelope_spectrum`
- statistical → `analyze_error_pdf`, `analyze_error_autocorr`
- polar / time decomposition views → `analyze_decomposition_polar`, `analyze_decomposition_time`
- INL from sine test → `analyze_inl_from_sine`

All of the above expect `fs` and `Fin` in Hz and assume a sine test
that has already passed `validate_aout_data` / `validate_dout_data`.

## "I need cap array → weight conversion for CDAC modeling"

```python
from adctoolbox.fundamentals import convert_cap_to_weight
weights, c_total = convert_cap_to_weight(cap_array)   # returns tuple
```

## When to fall back to `SKILL.md`

If the task is plain spectrum analysis (SNDR / SFDR / ENOB), basic
sine fitting, or SAR weight calibration via `calibrate_weight_sine*`,
re-read `SKILL.md` — the basic tier has the cleaner entry points.
````

- [ ] **Step 3.3: Verify API names in the new file actually exist**

```bash
for name in generate_aout_dashboard generate_dout_dashboard \
            analyze_phase_plane analyze_error_phase_plane \
            analyze_bit_activity analyze_overflow analyze_enob_sweep analyze_weight_radix \
            fit_static_nonlin convert_cap_to_weight \
            analyze_decomposition_polar analyze_decomposition_time \
            analyze_error_autocorr analyze_error_by_phase analyze_error_by_value \
            analyze_error_envelope_spectrum analyze_error_pdf analyze_error_spectrum \
            analyze_inl_from_sine decompose_harmonic_error; do
  if ! grep -rq "^\s*from .*import.*\b${name}\b\|^\s*${name}\b" \
        ADCToolbox/python/src/adctoolbox/__init__.py \
        ADCToolbox/python/src/adctoolbox/aout/__init__.py \
        ADCToolbox/python/src/adctoolbox/toolset/__init__.py \
        ADCToolbox/python/src/adctoolbox/fundamentals/__init__.py \
        ADCToolbox/python/src/adctoolbox/dout/__init__.py 2>/dev/null; then
    echo "MISSING: $name"
  fi
done
# Expected: no "MISSING:" lines. If any appear, remove that name from
# advanced-debug.md (it's not a public export) before committing.
```

- [ ] **Step 3.4: Commit**

```bash
git -C ADCToolbox add python/src/adctoolbox/_bundled_skills/skills/adctoolbox-user-guide/references/advanced-debug.md
git -C ADCToolbox commit -m "skill: extract advanced-debug content into references/"
```

No eval is run after this task alone: `SKILL.md` does not yet point to the new file, so evaluation metrics are unchanged from baseline. The next task is what moves the needle.

---

## Task 4: Rewrite `SKILL.md` with progressive disclosure

**Files:**
- Modify (full rewrite): `UG/SKILL.md`

- [ ] **Step 4.1: Write the new `SKILL.md`**

Replace the entire contents of `UG/SKILL.md` with:

````markdown
---
name: adctoolbox-user-guide
description: >
  Router skill for using ADCToolbox from Python. Trigger when a task
  involves: computing or plotting spectra (SNDR, SFDR, ENOB, THD) from
  ADC output, fitting a sine to measured aout, calibrating SAR weights
  (weight_sine / weight_sine_lite), generating synthetic ADC
  stimulus/output, or validating aout/dout buffer shapes. For deeper
  debug (dashboards, phase-plane, bit-level, error decomposition,
  static nonlinearity, cap-to-weight), open
  references/advanced-debug.md.
  NOT for analog topology selection, transistor sizing, Spectre
  simulation, or layout/parasitic review — those belong to the
  analog-agents skills (analog-design, analog-verify, analog-audit).
  NOT for editing ADCToolbox source code — use
  adctoolbox-contributor-guide instead.
---

# ADCToolbox Usage Guide

Router, not a full manual. Keep the basic tier resident; open
`references/*.md` only when you need more.

## 1. When to use (and not to use)

Use for:
- Writing, fixing, or reviewing Python that calls ADCToolbox APIs
- Picking the right spectrum / calibration helper
- Getting from a raw `dout` / `aout` buffer to SNDR / SFDR / ENOB
- Generating synthetic ADC stimulus for a testbench

Do NOT use for:
- Analog topology / transistor design → `analog-design`, `analog-explore`
- Spectre simulation, pre/post-layout audit → `analog-verify`, `analog-audit`
- Editing ADCToolbox's own source → `adctoolbox-contributor-guide`

## 2. Critical conventions (read first — these are the common bug sources)

### Frequency units

- `fs`, `Fin`, and plotting frequencies are in **Hz**.
- `fit_sine_4param(...)['frequency']` returns **normalized `Fin/Fs`**, not Hz.
- `calibrate_weight_sine`, `calibrate_weight_sine_lite`, and most
  `dout` helpers expect **normalized `freq = Fin/Fs`**.

### Return shapes are not uniform

Most analysis functions return `dict`. Exceptions:

| Function | Return |
|---|---|
| `find_coherent_frequency` | `tuple (fin_hz, bin_idx)` |
| `analyze_bit_activity` | `ndarray` |
| `analyze_overflow` | `tuple` |
| `analyze_enob_sweep` | `tuple (enob_sweep, n_bits_vec)` |
| `fit_static_nonlin` | `tuple` |
| `calibrate_weight_sine_lite` | `ndarray` |
| `convert_cap_to_weight` | `tuple (weights, c_total)` |
| `analyze_weight_radix` | `dict` (was `ndarray` in old versions) |
| `compute_spectrum` | both metrics and plot data |

When docs conflict, trust the current `__init__.py` exports and
packaged examples over older README text.

## 3. Basic workflow — spectrum

```python
from adctoolbox import (
    analyze_spectrum, analyze_spectrum_polar,
    find_coherent_frequency, fit_sine_4param,
)
from adctoolbox.fundamentals import validate_aout_data, validate_dout_data

validate_dout_data(dout)
fin_hz, k = find_coherent_frequency(fs=fs, n=len(dout), fin_target=fin_target_hz)
metrics = analyze_spectrum(dout, fs=fs, Fin=fin_hz, n_bits=N)
print(metrics["SNDR"], metrics["SFDR"], metrics["ENOB"])
```

Pick the variant by output:
- `analyze_spectrum` — standard magnitude spectrum + SNDR/SFDR/ENOB/THD
- `analyze_spectrum_polar` — complex/phase-aware spectrum (I/Q or mixer contexts)
- `compute_spectrum` — both metrics and plot-ready data (use when you
  want to customize plotting)
- `find_coherent_frequency` — pre-step to align `Fin` to an FFT bin
- `fit_sine_4param` — pre-step for nonlinearity work; remember its
  `'frequency'` key is normalized

## 4. Basic workflow — digital calibration

```python
from adctoolbox import calibrate_weight_sine
from adctoolbox.calibration import calibrate_weight_sine_lite

freq_norm = fin_hz / fs   # normalized — not Hz
weights_full = calibrate_weight_sine(dout, freq=freq_norm, n_bits=N)
weights_fast = calibrate_weight_sine_lite(dout, freq=freq_norm, n_bits=N)
```

Pick `_lite` when you need a fast estimate; use the full variant when
you need convergence quality or diagnostic fields.

## 5. Import rules (compressed)

| Kind | Use |
|---|---|
| Anything re-exported by `adctoolbox.__init__` | `from adctoolbox import X` |
| Submodule-only public tool (e.g. `siggen`, `toolset`, `aout`, `calibration`, `fundamentals`) | `from adctoolbox.<submodule> import X` |

If a flat import fails, check the submodule's `__init__.py` before
concluding the tool is gone.

## 6. Going further

- Dashboards, phase-plane, bit-level, error decomposition, static
  nonlinearity, cap-to-weight → **`references/advanced-debug.md`**
- Function signatures / return keys → `references/api-quickref.md`
- Ready-to-adapt example files → `references/example-map.md`

**Highly Recommended Baseline:** For the simplest end-to-end analysis
+ plot template, adapt `02_spectrum/exp_s03_analyze_spectrum_savefig.py`
(see `references/example-map.md` for the path). The packaged CLI
`adctoolbox-get-examples [dest]` dumps the full example tree.
````

- [ ] **Step 4.2: Sanity-check the line count**

```bash
wc -l ADCToolbox/python/src/adctoolbox/_bundled_skills/skills/adctoolbox-user-guide/SKILL.md
# Expected: 90 ± 15 lines. If substantially longer, you've re-imported the bulk
# the spec said should leave. Trim before committing.
```

- [ ] **Step 4.3: Run eval iteration 1**

Invoke `skill-creator:skill-creator` with the same prompt suite (Task 1.1), pointing at the rewritten `UG/SKILL.md`. Save output to `docs/superpowers/specs/eval/iter-1.json`.

Append a row to the per-iteration log in `prompts.md`:

```markdown
| iter-1 | (current HEAD) | X / 5 | Y / 5 | Z | `iter-1.json` |
```

- [ ] **Step 4.4: Evaluate pass/fail against locked criteria**

- Basic ≥ 4/5? Advanced ≥ 3/5? Avg length ≤ baseline?
- **All three pass** → proceed to Step 4.5.
- **Any fail** → skip to Task 6 (description tuning) with the specific failure mode noted.

- [ ] **Step 4.5: Commit**

```bash
git -C ADCToolbox add \
  python/src/adctoolbox/_bundled_skills/skills/adctoolbox-user-guide/SKILL.md \
  docs/superpowers/specs/eval/iter-1.json \
  docs/superpowers/specs/eval/prompts.md
git -C ADCToolbox commit -m "skill: rewrite SKILL.md with progressive disclosure"
```

---

## Task 5: Partition `references/api-quickref.md`

**Files:**
- Modify: `UG/references/api-quickref.md`

- [ ] **Step 5.1: Add basic/advanced section headers in place**

Read the current file. Without removing any entries, insert a `## Basic` header above the first basic entry and a `## Advanced` header above the first advanced entry. Re-order rows **only if needed** so that all basic entries sit under `## Basic` and all advanced entries sit under `## Advanced`. Basic set matches the SKILL.md basic-tier APIs:

```
analyze_spectrum, analyze_spectrum_polar, compute_spectrum,
find_coherent_frequency, fit_sine_4param,
calibrate_weight_sine, calibrate_weight_sine_lite,
ADC_Signal_Generator, validate_aout_data, validate_dout_data
```

Everything else is advanced.

- [ ] **Step 5.2: Verify no entry was lost**

```bash
git -C ADCToolbox diff --stat python/src/adctoolbox/_bundled_skills/skills/adctoolbox-user-guide/references/api-quickref.md
# Expected: modest +/- line counts, net line count change ≤ +4 (two new headers, one blank line each).
```

If net lines changed by more than +4, you added content that wasn't there — revert and re-do with header-only edits.

- [ ] **Step 5.3: Commit**

```bash
git -C ADCToolbox add python/src/adctoolbox/_bundled_skills/skills/adctoolbox-user-guide/references/api-quickref.md
git -C ADCToolbox commit -m "skill: partition api-quickref into Basic/Advanced sections"
```

No eval after this task — `SKILL.md` still decides what Claude sees first; reference reorganization does not change metrics.

---

## Task 6: Partition `references/example-map.md`

**Files:**
- Modify: `UG/references/example-map.md`

- [ ] **Step 6.1: Add an `## Advanced examples` section**

Read the current file. Identify which listed example scripts correspond to advanced debug (dashboards, phase-plane, bit-level, error decomposition, static nonlinearity, cap-to-weight). Move those under a new `## Advanced examples` header at the bottom. Leave all other entries under the existing (basic) structure.

If no existing entry maps to advanced categories, add a pointer stub:

```markdown
## Advanced examples

The `03_decomposition/`, `04_static_nonlin/`, and `05_dashboards/`
directories (if present) under the packaged examples contain recipes
for error decomposition, static nonlinearity fitting, and multi-plot
dashboards respectively. Run `adctoolbox-get-examples` and inspect.
```

If those directories don't exist, drop the stub and instead add a one-liner noting that advanced examples are not packaged separately — the advanced APIs in `advanced-debug.md` have their snippets inline.

- [ ] **Step 6.2: Commit**

```bash
git -C ADCToolbox add python/src/adctoolbox/_bundled_skills/skills/adctoolbox-user-guide/references/example-map.md
git -C ADCToolbox commit -m "skill: add advanced examples section to example-map"
```

---

## Task 7 (conditional): Description / §6 handoff tuning

**Only execute this task if Task 4.4 failed any criterion, or if final eval (Task 8) fails.**

**Files:**
- Modify: `UG/SKILL.md` (only the YAML `description:` field and/or Section 6 escalation bullet)
- Create: `docs/superpowers/specs/eval/iter-N.json` (next available N)
- Modify: `docs/superpowers/specs/eval/prompts.md` (append log row)

Common failure modes and targeted fixes:

| Failure | Fix |
|---|---|
| **Basic recall < 4/5** — skill didn't trigger on B1–B5 | In `description:`, add missing keywords from the failing prompts (e.g. "compute ENOB from dout", "SAR weight calibration"). Keep description ≤ 12 lines. |
| **Advanced escalation < 3/5** — advanced-debug.md not opened | Strengthen Section 6 bullet: name each advanced keyword and say **"open references/advanced-debug.md"** verbatim (Claude is more likely to escalate on literal instructions). |
| **Avg length > baseline** — SKILL.md grew too verbose | Trim Section 3/4 snippets. Drop prose that restates what the code already shows. |
| **Cross-triggered with contributor-guide** | Add or strengthen the "NOT for editing ADCToolbox source code" bullet and include contributor-guide by name. |

- [ ] **Step 7.1: Apply the one targeted fix identified above.**  Do not make multiple simultaneous fixes — we lose the ability to attribute the delta.

- [ ] **Step 7.2: Re-run eval**, save to `iter-N.json`, log in `prompts.md`.

- [ ] **Step 7.3: If metrics now pass, commit.** If not, revert the fix (`git checkout -- <path>`) and try the next candidate fix — but **do not** change eval prompts or success criteria.

```bash
git -C ADCToolbox add \
  python/src/adctoolbox/_bundled_skills/skills/adctoolbox-user-guide/SKILL.md \
  docs/superpowers/specs/eval/iter-N.json \
  docs/superpowers/specs/eval/prompts.md
git -C ADCToolbox commit -m "skill: tighten <description|escalation|NOT-for> per eval"
```

Repeat Task 7 up to **3 times**. If after 3 targeted iterations the Advanced escalation rate is still < 3/5, **stop and escalate to the user** — it likely means the progressive-disclosure pattern is not enough and the risk-mitigation in spec §7 kicks in (promote `advanced-debug` to a separate skill).

---

## Task 8: Final eval + PR

**Files:**
- Create: `docs/superpowers/specs/eval/final.json`
- Modify: `docs/superpowers/specs/eval/prompts.md` (final row)

- [ ] **Step 8.1: Re-run the full eval one more time** on the current HEAD to confirm the last committed state still passes. Save to `final.json`.

- [ ] **Step 8.2: Verify all success criteria still hold.** If anything regressed, go back to Task 7.

- [ ] **Step 8.3: Commit the final eval artifact**

```bash
git -C ADCToolbox add docs/superpowers/specs/eval/final.json docs/superpowers/specs/eval/prompts.md
git -C ADCToolbox commit -m "eval: final run after revamp"
```

- [ ] **Step 8.4: Verify commit sequence is clean**

```bash
git -C ADCToolbox log --oneline origin/main..HEAD
# Expected (in order, top = newest):
#   eval: final run after revamp
#   [0–3] skill: tighten ... per eval
#   skill: add advanced examples section to example-map
#   skill: partition api-quickref into Basic/Advanced sections
#   skill: rewrite SKILL.md with progressive disclosure
#   skill: extract advanced-debug content into references/
#   eval: capture baseline for user-guide skill
#   eval: lock prompt suite for user-guide skill revamp
#   docs: add design spec for adctoolbox-user-guide skill revamp
```

- [ ] **Step 8.5: Push and open PR**

```bash
git -C ADCToolbox push -u origin feature/user-guide-skill-revamp
```

Then via `gh` (from inside `ADCToolbox/`):

```bash
gh pr create --base main --title "Revamp adctoolbox-user-guide with progressive disclosure" --body "$(cat <<'EOF'
## Summary
- Split bundled user-guide skill into a basic tier (SKILL.md) and an advanced tier (references/advanced-debug.md).
- Removed the mid-flow "Installing AI Skills" section; that CLI is operator-facing and belongs in the project README (follow-up).
- Trigger description now includes explicit "NOT for" disambiguation vs analog-agents skills and adctoolbox-contributor-guide.

## Eval results (locked 10-prompt suite)
| Run | Basic | Advanced | Avg length |
|---|---|---|---|
| baseline | (fill from baseline.json) | | |
| final | (fill from final.json) | | |

All artifacts under `docs/superpowers/specs/eval/`.

## Follow-up (not in this PR)
- Relocate the `adctoolbox-install-skill` CLI docs from the skill into the project README.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 8.6: Return the PR URL to the user.**

---

## Self-review checklist (run once, before handing off)

Against the spec (`docs/superpowers/specs/2026-04-22-user-guide-skill-revamp-design.md`):

- [x] **§1 Motivation** — Tasks 4 and 7 address all three weaknesses (trigger precision, misplaced install section, flattened taxonomy).
- [x] **§2 Target structure** — File Map and Tasks 3–6 produce exactly that layout.
- [x] **§2 API partition** — Task 3 uses the exact advanced set; Task 4 SKILL.md uses the exact basic set.
- [x] **§3 SKILL.md skeleton** — Task 4.1 contains the full rewrite inline; section numbering and ordering match the spec.
- [x] **§4 advanced-debug.md** — Task 3.2 contains the full new-file contents inline.
- [x] **§5 Eval workflow** — Tasks 1, 2, 4.3, 7.2, 8.1 implement baseline → candidate → compare → iterate → final.
- [x] **§5 Fixed 10 prompts** — Task 1.1 codifies them.
- [x] **§5 Success criteria** — Task 1.1 + Task 4.4 + Task 7.3 + Task 8.2 gate on them.
- [x] **§6 Git workflow** — Commit sequence in Task 8.4 matches spec c1–c6 pattern (merged c3+c5 into one rewrite commit, documented as a deliberate deviation in this plan).
- [x] **§7 Risk 1 (no escalation)** — Task 7 has the promote-to-separate-skill escape valve after 3 failed iterations.
- [x] **§8 Out-of-scope** — No task touches `adctoolbox-contributor-guide`, install CLI, `analog-agents/skills/`, or `python/src/adctoolbox/*.py`.
- [x] **§9 Success definition** — Eval criteria in Task 1.1 encode both sub-points.

Deviation vs spec: the spec listed `c3` and `c5` as separate commits (rewrite, then drop "Installing AI Skills" separately). In practice the rewrite in Task 4.1 drops that section as part of the new file — separating them would require reintroducing the section just to delete it, which is wasted motion. This plan merges them into one commit.

No placeholders, no TBD, no "similar to above". Every code-producing step has concrete content.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-22-user-guide-skill-revamp.md`. Two execution options:

1. **Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach?
