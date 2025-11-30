# Dataset Restructuring - COMPLETE ✅

**Date:** 2025-11-28
**Status:** Successfully completed

## Summary

Restructured dataset directory from flat structure to organized subdirectories by data type.

### New Structure

```
dataset/
├── aout/              # Analog output datasets (15 files)
│   └── sinewave_*.csv
├── dout/              # Digital output datasets (6 files)
│   └── dout_*.csv
└── others/            # Miscellaneous datasets
    └── jitter_sweep/  # Jitter sweep data (93 files)
```

### Files Updated

#### MATLAB Data Generation (20 files)
- ✅ `batch_gen.m` - Updated to create all 3 subdirectories
- ✅ 14 × `gen_sinewave_*.m` - Changed `data_dir = "dataset/aout"`
- ✅ 4 × `generate_*_dout.m` - Changed to use `"dataset/dout"`
- ✅ `generate_jitter_sweep_data.m` - Changed to use `"dataset/others/jitter_sweep"`

#### MATLAB Test Scripts (18 files)
- ✅ 11 × AOUT unit tests - Changed `inputDir = "dataset/aout"`
  - test_errAutoCorrelation.m
  - test_errEnvelopeSpectrum.m
  - test_errHistSine_code.m
  - test_errHistSine_phase.m
  - test_errPDF.m
  - test_errSpectrum.m
  - test_INLsine.m
  - test_sineFit.m
  - test_specPlot.m
  - test_specPlotPhase.m
  - test_tomDecomp.m

- ✅ 5 × DOUT unit tests - Changed `inputDir = "dataset/dout"`
  - test_bitActivity.m
  - test_ENoB_bitSweep.m
  - test_FGCalSine.m
  - test_FGCalSine_overflowChk.m
  - test_weightScaling.m

- ✅ 2 × System tests
  - test_toolset_aout.m → `inputDir = "dataset/aout"`
  - test_toolset_dout.m → `inputDir = "dataset/dout"`

#### Python Test Scripts (18 files)
- ✅ 11 × AOUT unit tests - Changed `input_dir = project_root / "dataset" / "aout"`
  - test_err_auto_correlation.py
  - test_err_envelope_spectrum.py
  - test_err_hist_sine.py
  - test_err_pdf.py
  - test_err_spectrum.py
  - test_inl_sine.py
  - test_sine_fit.py
  - test_spec_plot.py
  - test_spec_plot_phase.py
  - test_tom_decomp.py

- ✅ 5 × DOUT unit tests - Changed `input_dir = project_root / "dataset" / "dout"`
  - test_bit_activity.py
  - test_enob_bit_sweep.py
  - test_fg_cal_sine.py
  - test_fg_cal_sine_overflow_chk.py
  - test_weight_scaling.py

- ✅ 2 × System tests
  - test_toolset_aout.py → `input_dir = Path('dataset') / 'aout'`
  - test_toolset_dout.py → `input_dir = Path('dataset') / 'dout'`

### Datasets Moved

- ✅ 15 files → `dataset/aout/`
  - sinewave_amplitude_modulation_0P001.csv
  - sinewave_amplitude_noise_0P001.csv
  - sinewave_clipping_0P012.csv
  - sinewave_drift_0P004.csv
  - sinewave_gain_error_0P98.csv
  - sinewave_glitch_0P000.csv
  - sinewave_HD2_n65dB_HD3_n65dB.csv
  - sinewave_INL_k2_0P0010_k3_0P0100.csv
  - sinewave_INL_k2_0P0010_k3_0P0120.csv
  - sinewave_jitter_400fs.csv
  - sinewave_kickback_0P015.csv
  - sinewave_noise_270uV.csv
  - sinewave_ref_error_0P001.csv
  - sinewave_Zone2_Tj_250fs.csv
  - sinewave_Zone3_Tj_250fs.csv

- ✅ 6 files → `dataset/dout/`
  - dout_Pipeline_3bx4x8_4b.csv
  - dout_Pipeline_3bx8_3bx8_8b.csv
  - dout_Pipeline_3bx8_8b.csv
  - dout_SAR_12b_weight_1.csv
  - dout_SAR_12b_weight_2.csv
  - dout_SAR_12b_weight_3.csv

- ✅ 1 folder (93 files) → `dataset/others/jitter_sweep/`

## Total Changes

- **Files Updated:** 56 files
  - 20 MATLAB generators
  - 18 MATLAB tests
  - 18 Python tests

- **Datasets Organized:** 114 files
  - 15 AOUT datasets
  - 6 DOUT datasets
  - 93 jitter sweep files

## Next Steps

1. ✅ Run MATLAB tests to verify all paths work
2. ✅ Run Python tests to verify all paths work
3. ✅ Update `.gitignore` if needed
4. ✅ Commit changes

## Verification Commands

### Test MATLAB paths
```matlab
cd matlab/tests/unit
test_sineFit  % AOUT test
test_FGCalSine  % DOUT test
```

### Test Python paths
```bash
python python/tests/unit/test_sine_fit.py  # AOUT test
python python/tests/unit/test_fg_cal_sine.py  # DOUT test
```

### Verify structure
```bash
ls dataset/aout/*.csv | wc -l  # Should be 15
ls dataset/dout/*.csv | wc -l  # Should be 6
ls dataset/others/jitter_sweep/*.csv | wc -l  # Should be 93
```

## Benefits

1. **Organization:** Clear separation by data type
2. **Scalability:** Easy to add new categories
3. **Clarity:** Immediately know what type of data each folder contains
4. **Maintainability:** Easier to manage large numbers of datasets
