# Supervised foreground calibration

These examples fit a digital output-to-reference mapping and freeze it for normal
operation. Start with [theory and scope](FOREGROUND_THEORY.md), then
[workflows](FOREGROUND_WORKFLOWS.md) and [current results](FOREGROUND_REPOSITORY_RESULTS.md).

- `exp_cal03_foreground_nonlinear.py`: original repository nonlinearity and memory models; polynomial/MLP correction.
- `exp_cal05_foreground_excitation.py`: matched-budget single versus mixed excitation; add `--retrain-mlp`.
- `audit_foreground_transfer.py`: frozen random two/three/five-tone transfer tests.
- `exp_cal10_foreground_bitweights.py`: [SNDR/SFDR and bit-weight comparisons](FOREGROUND_BITWEIGHTS.md).
- `exp_cal11_combined_foreground.py`: [combined mismatch and nonlinearity](COMBINED_FOREGROUND.md), including cascaded corrections.
- `audit_repository_foreground.py`, `audit_foreground_bitweights.py`, `audit_combined_foreground.py`: independent frozen replay checks.

The original experiment numbers are retained. This directory has no dependency
on the separate Dither examples. Public fitting APIs are in `adctoolbox.calibration`.
References, input units, alignment, training coverage and calibration access are
part of each protocol. These results are behavioral, not measured silicon.

The earlier custom FIR converter was removed before this extraction. Historical
evidence keeps its original paths and source hashes; use its matching snapshot
for exact replay. Current manifests distinguish the repository model explicitly.
Use fresh output directories; existing results are never overwritten.
