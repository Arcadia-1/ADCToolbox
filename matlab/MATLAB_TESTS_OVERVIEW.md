# MATLAB Test Workflow

The MATLAB tests are an optional external validation suite. They require a
licensed MATLAB installation and are not part of the default Python pytest
workflow.

## Requirements

- MATLAB R2020a or newer
- Signal Processing Toolbox
- A headless-capable display setup for figure generation, or a MATLAB
  environment that can save figures without opening desktop windows

## Test Entrypoint

Use the Python wrapper from the repository root:

```bash
python matlab/tests/run_matlab_tests.py all
```

The wrapper locates MATLAB in this order:

1. `--matlab-executable <path>`
2. `ADCTOOLBOX_MATLAB`
3. `MATLAB_EXECUTABLE`
4. `PATH`
5. Common MATLAB install folders

When `--matlab-executable` is provided, it is strict: if that path or command
cannot be resolved, the wrapper exits with code `2` instead of silently falling
back to another MATLAB installation.

Available suites:

```bash
python matlab/tests/run_matlab_tests.py common
python matlab/tests/run_matlab_tests.py aout
python matlab/tests/run_matlab_tests.py dout
python matlab/tests/run_matlab_tests.py all
python matlab/tests/run_matlab_tests.py jitter
```

The `all` suite is the default smoke suite for committed data. It runs
`common`, `aout`, and `dout`. The jitter sweep is generated data, so `all`
prints a skip message when `matlab/test_dataset/jitter_sweep/config.csv` is
absent. Run `jitter` explicitly after generating the sweep data with
`matlab/data_generation/gen_jitter_sweep_data.m`.

For CI environments where MATLAB is optional, use:

```bash
python matlab/tests/run_matlab_tests.py all --missing-ok
```

## Exit Codes

| Exit code | Meaning |
|---|---|
| `0` | MATLAB was available and the selected suite passed, or `--missing-ok` allowed a missing MATLAB install |
| `2` | `--matlab-executable` was provided but did not resolve to an executable |
| `77` | MATLAB was not found; optional external test environment is unavailable |
| other nonzero | MATLAB launched but the selected suite failed |

## Output Locations

The MATLAB scripts change into the `matlab/` root before running. Generated
outputs are written under:

- `matlab/test_output/`
- `matlab/test_plots/`

These are generated artifacts and should not be committed.

Committed MATLAB golden CSVs used by Python comparison tests live under:

- `reference_output/`

Input datasets are discovered from `matlab/test_dataset/` when present. If that
local generated-data folder does not exist, the committed-data MATLAB tests fall
back to the repository-level `reference_dataset/` folder. Generated-data suites
such as `jitter` still require their generated inputs under
`matlab/test_dataset/`.

## Direct MATLAB Invocation

On a machine with MATLAB installed, the wrapper runs this shape of command:

```bash
matlab -batch "cd('path/to/ADCToolbox/matlab'); addpath(genpath(fullfile(pwd,'src'))); addpath(genpath(fullfile(pwd,'tests'))); run_all"
```

You can still run the scripts directly inside MATLAB:

```matlab
cd matlab/tests
run_all
```

The wrapper exists to make missing MATLAB environments explicit instead of
failing later with an unclear shell or path error.
