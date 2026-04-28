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
| baseline  | 687dbf1 (pre-change) | 5 / 5 | 0 / 5 | 200.7 | `baseline.json` |
