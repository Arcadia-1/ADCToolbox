# Repository-model foreground results

The custom `NonlinearFrontend` converter has been removed. `exp_cal03/05` now
call the original `ADC_Signal_Generator.apply_static_nonlinearity`,
`apply_memory_effect` and `apply_quantization_noise`. See packaged
`FOREGROUND_WORKFLOWS.md` for exact composition,
coarse/fine quantization, memory wrap handling, references and reproduction.

The acceptance rerun uses 400 training and 40 validation captures of 8192 points,
stride 16 (204,800 / 20,480 rows), seeds 11/22/33, and 200 MLP iterations.
Single-tone and mixed-excitation models are retrained; mixed means half single,
half dual tone. All methods use identical held-out captures. Fifteen single-tone
amplitude/frequency conditions are averaged within seed, then across three seeds.
Each cell below is **SNDR / SFDR in dB**. These are behavioral results, not silicon.

| Training / correction | Static model | Coarse-feedback memory model |
| --- | ---: | ---: |
| Raw | 49.43 / 51.39 | 42.50 / 45.71 |
| Static poly5 / mixed | 76.64 / 98.32 | 47.81 / 53.51 |
| History poly5 / single | 76.69 / 96.62 | 51.20 / 60.46 |
| History poly5 / mixed | 76.62 / 98.10 | 51.02 / 60.15 |
| MLP / single | 68.75 / 74.58 | 50.96 / 60.18 |
| MLP / mixed | 69.33 / 77.73 | 50.84 / 59.89 |

The independent frozen replay checks 2904 rows, including
72 fresh random 2/3/5-tone captures; full-complex FFT agreement is within
1.48e-11 dB. Multi-tone ratios are reported
separately, not called single-tone SNDR/SFDR. Run `audit_repository_foreground.py`
with `--baseline`, `--mixed`, `--fresh` and a new `--output` JSON path.

Of 12 MLP fits, 0 satisfy optimizer convergence. An iteration limit is
not convergence; independent validation selects the saved iterate. The original
memory API includes coarse/fine quantization and discontinuous code feedback,
so the removed smooth FIR scores do not apply. The nominal input gain also
changed from .99 to 1 to use the original static API directly. Fixed device
parameters across seeds do not constitute device Monte Carlo.

These models require foreground reference captures and frozen inference under
matched operating conditions. Reusing repository code does not prove physical
fidelity or remove the favorable relationship between a smooth polynomial plant
and polynomial correction. See packaged `FOREGROUND_THEORY.md` for physical evidence.
Old artifacts retain their source hashes and require their historical code;
current workflows reject old manifests rather than relabeling their results.

## Multi-tone generalization

Frozen tests use 72 random captures (two plants, three seeds, 2/3/5 tones,
four captures each), excluding training and validation frequencies. Values below
are absolute-error SNR in dB, without gain/phase/DC alignment; they are not SNDR.
History models use the current sample and three preceding samples.

| Plant / tones | Raw | Single-trained history poly5 | Mixed-trained history poly5 | Mixed-trained MLP |
| --- | ---: | ---: | ---: | ---: |
| Static / 2 | 40.08 | 42.77 | 78.14 | 70.86 |
| Static / 3 | 43.82 | 42.85 | 75.44 | 70.27 |
| Static / 5 | 45.85 | 41.57 | 73.33 | 69.05 |
| Memory / 2 | 27.71 | 32.59 | 52.25 | 52.19 |
| Memory / 3 | 27.67 | 31.75 | 49.49 | 49.46 |
| Memory / 5 | 27.34 | 30.89 | 47.41 | 47.39 |

Single-tone test performance does not establish multi-tone generalization.
The repository memory model's quantization and code feedback remain limiting
for the tested polynomial and MLP structures; the old smooth-FIR scores do not apply.

## Independent SAR mismatch comparison

These cases contain the specified 1% MSB mismatch without the smooth frontend.
They use 64 mixed training / 8 validation captures, 8192 samples, stride 8,
seeds 11/22/33, and 1000 MLP iterations. Values are mean SNDR / SFDR (dB)
over the same 15 single-tone conditions per seed. Bit methods observe internal
SAR decisions; polynomial/MLP methods observe reconstructed outputs. The sine
bit solver uses a separate full-record protocol. See `FOREGROUND_BITWEIGHTS.md`.

| Method | Binary SAR | Redundant SAR |
| --- | ---: | ---: |
| Raw | 41.05 / 44.12 | 53.71 / 65.01 |
| Sine bit weights | 53.17 / 71.28 | 77.54 / 102.69 |
| Reference bit weights | 53.02 / 68.80 | 77.54 / 102.74 |
| Static polynomial | 42.13 / 46.47 | 52.01 / 56.67 |
| History polynomial | 40.04 / 43.69 | 50.47 / 54.58 |
| MLP | 55.57 / 70.65 | 51.95 / 67.16 |

## Combined mismatch and nonlinearity

Both errors are present in the same device. Default budgets are 64 mixed training
/ 8 validation captures, 8192 samples, stride 8, three seeds and 1000 MLP iterations.
Each cell is mean SNDR / SFDR (dB) across 15 single-tone conditions per seed.
The cascade requires internal decisions and an additional 8192-sample calibration
capture injected at the SAR input, bypassing frontend nonlinearity. See
`COMBINED_FOREGROUND.md` for the hardware-access assumption and failure controls.

| Method | Binary SAR | Redundant SAR |
| --- | ---: | ---: |
| Raw | 36.61 / 38.91 | 46.75 / 49.22 |
| Bit weights | 41.25 / 46.62 | 49.51 / 51.37 |
| Polynomial | 42.84 / 47.27 | 52.81 / 57.43 |
| MLP | 54.36 / 68.38 | 51.71 / 65.15 |
| Bit weights + polynomial | 53.16 / 69.23 | 77.66 / 99.92 |
| Bit weights + MLP | 53.27 / 68.90 | 76.05 / 90.56 |

All 12 combined MLP fits reached the iteration limit; saved iterates were selected
by independent validation. The cascades substantially improve the redundant SAR
case, but do not universally beat standalone correction on binary SAR. These
tables summarize the stated experiments, not architecture-independent guarantees.

## Reproducing detailed results

The scripts generate capture manifests, frozen models, loss histories, per-test
CSV metrics, plots and source hashes under the requested output directory.
Run the adjacent audit scripts to check the generated outputs. Generated records,
local installation reports and retired custom-FIR results are not source assets
and are intentionally not committed here. The current summary tables and runnable
protocols are retained; old results are not relabeled as current-model evidence.
