%% Batch Generation - Run all sinewave generators
% This script runs all individual generator scripts in sequence
% Comment out any generators you don't need

close all; clear; clc; warning("off");

fprintf("\n=== Generating AOUT datasets ===\n");
gen_sinewave_jitter
gen_sinewave_clipping
gen_sinewave_gain_error
gen_sinewave_kickback
gen_sinewave_glitch
gen_sinewave_drift
gen_sinewave_ref_error
gen_sinewave_noise
gen_sinewave_nonlin_HD
gen_sinewave_nonlin_static
gen_sinewave_nonlin_dynamic
gen_sinewave_am_tone
gen_sinewave_am_noise
gen_sinewave_nyquist_zones
gen_sinewave_multirun

fprintf("\n=== Generating OTHER datasets ===\n");
% generate_jitter_sweep_data

fprintf("\n=== Generating DOUT datasets ===\n");
generate_pipeline2s_dout
generate_pipeline3s_dout
generate_pipeline8s_dout
generate_sar_dout

fprintf("\n=== Batch generation complete ===\n");