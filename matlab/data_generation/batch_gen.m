%% Batch Generation - Run all sinewave generators
% This script runs all individual generator scripts in sequence
% Comment out any generators you don't need

run(fullfile(fileparts(mfilename('fullpath')), '..', 'tests', 'utils', 'ensureMatlabRoot.m'));
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
gen_sinewave_nonlin_hd
gen_sinewave_nonlin_static
gen_sinewave_nonlin_dynamic
gen_sinewave_am_tone
gen_sinewave_am_noise
gen_sinewave_nyquist_zones
gen_sinewave_multirun

fprintf("\n=== Generating OTHER datasets ===\n");
% gen_jitter_sweep_data

fprintf("\n=== Generating DOUT datasets ===\n");
gen_pipeline2s_dout
gen_pipeline3s_dout
gen_pipeline8s_dout
gen_sar_dout

fprintf("\n=== Batch generation complete ===\n");
