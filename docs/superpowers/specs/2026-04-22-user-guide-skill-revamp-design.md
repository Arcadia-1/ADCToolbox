# Design: `adctoolbox-user-guide` Skill Revamp

**Date:** 2026-04-22
**Branch:** `feature/user-guide-skill-revamp` (ADCToolbox)
**Scope:** Rewrite the bundled `adctoolbox-user-guide` skill using
Progressive Disclosure (basic tier in `SKILL.md`, advanced tier in
`references/advanced-debug.md`). Drive the rewrite with `skill-creator`
evals (baseline → candidate → compare) so every commit is justified by
measured behavior change, not taste.

---

## 1. Motivation

Current `SKILL.md` (129 lines) has three measurable weaknesses in how
users' agents interact with ADCToolbox:

- **Trigger is a comma-list union** → high recall, low precision; no
  "when NOT to use" to protect against misfires.
- **"Installing AI Skills" meta-section sits mid-flow** → pollutes the
  basic-task path with install-time noise.
- **Basic workflow and deep-debug workflow are flattened into one
  taxonomy** → model has to scan over advanced debug API names
  (`analyze_phase_plane`, `analyze_bit_activity`, …) even when the task
  is a simple "plot a spectrum / compute SNDR".

User goal: the skill should *correctly teach* the user's agent how to
use ADCToolbox, with a basic tier (spectrum + calibration) that is
always loaded and an advanced tier (debug tools) that loads on demand.

Out of scope for this revamp:
- `adctoolbox-contributor-guide` (not touched)
- Python source under `python/src/adctoolbox/`
- `adctoolbox-install-skill` CLI flags (still a single skill, no
  `--basic`/`--advanced`)
- Any skill under `analog-agents/skills/` (including `adc-analyzer`)

---

## 2. Target Structure

```
_bundled_skills/skills/adctoolbox-user-guide/
├── SKILL.md                          # Basic tier: spectrum + calibration + conventions
└── references/
    ├── api-quickref.md               # Existing; re-partitioned into Basic / Advanced sections
    ├── example-map.md                # Existing; adds an "Advanced examples" section
    └── advanced-debug.md             # New: dashboards / phase-plane / bit-level / error decomp
```

### API partition (locked)

**Basic → `SKILL.md`:**

| Category           | Functions |
|--------------------|-----------|
| Spectrum analysis  | `analyze_spectrum`, `analyze_spectrum_polar`, `find_coherent_frequency`, `compute_spectrum` |
| Sine fitting       | `fit_sine_4param` (primitive for basic flow; also the source of the `Fin/Fs` normalized-frequency trap — kept adjacent to the convention section) |
| Digital calibration| `calibrate_weight_sine`, `calibrate_weight_sine_lite` |
| Signal generation  | `ADC_Signal_Generator` |
| Input validation   | `validate_aout_data`, `validate_dout_data` |

**Advanced → `references/advanced-debug.md`:**

| Category                | Functions |
|-------------------------|-----------|
| Dashboards (multi-plot) | `generate_aout_dashboard`, `generate_dout_dashboard` |
| Phase-plane debug       | `analyze_phase_plane`, `analyze_error_phase_plane` |
| Bit-level analysis      | `analyze_bit_activity`, `analyze_overflow`, `analyze_enob_sweep`, `analyze_weight_radix` |
| Static nonlinearity     | `fit_static_nonlin` |
| Error decomposition     | error-analysis helpers, decomposition helpers |
| Unit conversions        | `convert_cap_to_weight` |

---

## 3. New `SKILL.md` Skeleton (target 70–90 lines)

```
---
name: adctoolbox-user-guide
description: >
  Router skill for using ADCToolbox from Python. Trigger when a task
  involves: plotting/computing spectra (SNDR, SFDR, ENOB, THD) from
  ADC output, fitting a sine to measured aout, calibrating SAR weights
  (weight_sine / weight_sine_lite), generating synthetic ADC
  stimulus/output, or validating aout/dout buffer shapes. For deeper
  debug (dashboards, phase-plane, bit activity, overflow, error
  decomposition) escalate to references/advanced-debug.md after
  loading.
  NOT for: analog topology selection, transistor sizing, Spectre
  simulation, or layout/parasitic review — those are handled by the
  analog-agents skills (analog-design, analog-verify, analog-audit).
  NOT for: editing ADCToolbox source code — use
  adctoolbox-contributor-guide instead.
---

# ADCToolbox Usage Guide

[One-line mission statement]

## 1. When to use / NOT to use
- use / NOT use bullets (explicit disambiguation vs contributor guide and analog-agents skills)

## 2. Critical conventions (READ FIRST — these are the common bug sources)
- Frequency units: `fs`, `Fin`, plotting freqs in Hz;
  `fit_sine_4param(...)['frequency']` is normalized `Fin/Fs`;
  `calibrate_weight_sine*` and most dout helpers expect normalized `freq = Fin/Fs`.
- Return shapes are not uniform (compact table, unchanged from current).

## 3. Basic workflow — spectrum
- Import lines (flat)
- One 10-line snippet: dout → SNDR/SFDR/ENOB using analyze_spectrum
- Brief note: when to pick analyze_spectrum_polar / compute_spectrum / find_coherent_frequency

## 4. Basic workflow — digital calibration
- Import lines
- One snippet for calibrate_weight_sine (and when to pick _lite)

## 5. Import rules (compressed from current Section 3)
- Flat for exported symbols; submodule for rest; ≤ 3 lines + one example

## 6. Going further
- Advanced debug (dashboards, phase-plane, bit-level, error decomp, cap-to-weight)
  → open references/advanced-debug.md
- Function signatures / return keys → references/api-quickref.md
- Ready-to-adapt example files → references/example-map.md
```

Sections removed from the current file:
- "Installing AI Skills" (the `adctoolbox-install-skill` CLI doc) — it
  is operator-facing, not model-facing, and blocks the main workflow.
  Will live in the project README (not in this revamp's scope; flag in
  the PR description).

---

## 4. New `references/advanced-debug.md`

Organized by **task keyword** (what the user asks), not by category:

```
# ADCToolbox — Advanced Debug Reference

Load this file when the agent needs deeper post-analysis than the
basic spectrum/calibration flow provides.

## "I want a single summary image of all aout/dout characteristics"
  → generate_aout_dashboard, generate_dout_dashboard
    + snippet + pointer to example-map.md entry

## "I need to see nonlinearity structure, not just a number"
  → analyze_phase_plane, analyze_error_phase_plane
    + when each applies + snippet

## "I want to look at what each ADC bit is doing"
  → analyze_bit_activity, analyze_overflow, analyze_enob_sweep, analyze_weight_radix
    + one paragraph per function + return-shape reminder

## "I have static INL/DNL data and want a nonlinearity fit"
  → fit_static_nonlin

## "I want to decompose total error into components"
  → error-analysis helpers, decomposition helpers
    (exact public surface to be enumerated against __init__.py during implementation)

## "I need cap array → weight conversion for CDAC modeling"
  → convert_cap_to_weight
```

---

## 5. `skill-creator` Eval Workflow

Each iteration follows this loop:

| Phase       | Action                                                                                     | Artifact              |
|-------------|--------------------------------------------------------------------------------------------|-----------------------|
| **Baseline** | Run 10 fixed prompts against the **current** `SKILL.md` (before any edits in this branch) | `eval-baseline.json`  |
| **Candidate**| Run the same 10 prompts against the rewritten skill                                       | `eval-candidate.json` |
| **Compare**  | `skill-creator` variance analysis on trigger rate + API-name correctness + length         | Pass / iterate / revert |
| **Iterate**  | Targeted edit (usually description or §6 handoff wording), re-run from Candidate          | next commit           |

### Fixed eval prompts (locked before baseline)

**Basic (5):**
1. "Help me compute SNDR from this ADC dout array."
2. "Plot a spectrum of the ADC output with coherent sampling."
3. "Go from dout to ENOB — give me the code."
4. "Calibrate SAR capacitor weights from a sine test."
5. "Generate a synthetic ADC output with a sinusoidal stimulus for testing."

**Advanced (5):**
1. "Plot a phase-plane to diagnose nonlinearity in the ADC aout."
2. "Show me per-bit activity for an N-bit ADC code stream."
3. "How do I detect overflow events in dout?"
4. "Give me a one-shot dashboard of all aout diagnostics."
5. "Convert a binary-weighted cap array to normalized weights."

### Success criteria (locked before baseline)

- **Basic:** ≥ 4/5 prompts where (a) the skill triggers, and (b) the
  response references the correct top-level API name.
- **Advanced:** ≥ 3/5 prompts where the response explicitly escalates
  to `references/advanced-debug.md` (i.e., loads or points to it).
- **Average response length:** non-increasing vs baseline (the revamp
  must not become more verbose to hit the other metrics).

A candidate commit is kept only when all three hold. If it fails, the
commit is either iterated in-place or reverted.

Eval artifacts live in `docs/superpowers/specs/eval/` (per iteration,
timestamped) and are **not** shipped in the pip package.

---

## 6. Git & PR Workflow

```
# Already done: branched off origin/main
git checkout feature/user-guide-skill-revamp

# Staged commits (each independently reviewable / revertable):
c1  eval: capture baseline for user-guide skill
c2  skill: extract advanced-debug content into references/
c3  skill: rewrite SKILL.md with progressive disclosure (basic tier)
c4  skill: tighten trigger description per eval results
c5  skill: drop "Installing AI Skills" section
c6+ skill: iteration fixes driven by eval deltas

git push -u origin feature/user-guide-skill-revamp
gh pr create --base main
```

The PR description notes the eval numbers (baseline vs final) and
flags the README work (install CLI doc relocation) as a follow-up.

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Model does not escalate to `advanced-debug.md` — advanced tools become invisible | Advanced-5 eval measures escalation rate directly. If ≤ 2/5, promote to a separate `adctoolbox-debug` skill (structurally option 3 from brainstorming). |
| Baseline eval shows the current skill is already close to ceiling; revamp adds noise | Each commit is independently revertable; final eval gates the PR merge. |
| Contributor-guide skill ends up cross-triggered with user-guide | `SKILL.md` §1 has explicit "NOT for" bullet pointing to contributor-guide; advanced-5 evals implicitly check this. |
| Example references (`example-map.md`) drift during the split | Implementation step includes grep for example filenames against repo state before shipping. |

---

## 8. Explicitly Out of Scope (Re-statement)

- Touching `adctoolbox-contributor-guide`
- Modifying `adctoolbox-install-skill` CLI (no `--basic` / `--advanced`)
- Any skill outside `_bundled_skills/` (nothing in `analog-agents/skills/`)
- Any change under `python/src/adctoolbox/` (this revamp is text-only)

---

## 9. Success Definition (One-Liner)

A user whose agent installs only `adctoolbox-user-guide` should:
1. Get correct spectrum + calibration code on a basic prompt, without
   the model hallucinating API names, and
2. Be told to open `references/advanced-debug.md` when the prompt
   involves phase-plane / bit-level / dashboard / error-decomposition
   territory — and on open, receive correct API guidance there.

Both measured by the eval suite in §5.
