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
