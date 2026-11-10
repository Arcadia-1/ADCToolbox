# Oversampling Examples

These examples demonstrate oversampling workflows. Examples 01–03 use
MATLAB-compatible analysis helpers; example 04 adds Python weight calibration.

- `exp_o01_noise_shaping_spectrum.py` - Generate a noise-shaped signal and analyze it with OSR.
- `exp_o02_ifilter_band_analysis.py` - Use `ifilter` to isolate the signal band.
- `exp_o03_ntfperf_perfosr.py` - Compare theoretical NTF gain with a performance-vs-OSR sweep.
- `exp_o04_inband_weight_calibration.py` - Compare full-band and in-band weight
  fitting on a 6-bit SAR with ideal second-order residue feedback and mismatched
  DAC weights. Freeze weights and evaluate a different capture. Saves full-band
  and zoomed spectra, weight errors, and JSON metrics. Use `--output DIR` to
  choose an output directory.

