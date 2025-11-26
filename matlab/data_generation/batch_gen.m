%% Batch Generation - Run all sinewave generators
% This script runs all individual generator scripts in sequence
% Comment out any generators you don't need

close all; clear; clc; warning("off");
rng(42); % Set random seed for reproducibility

% Create output directory
data_dir = "dataset";
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
    fprintf("Created output directory: [%s]\n", data_dir);
end

gen_sinewave_jitter
gen_sinewave_clipping
gen_sinewave_gain_error
gen_sinewave_kickback
gen_sinewave_glitch
gen_sinewave_drift
gen_sinewave_ref_error
gen_sinewave_noise
gen_sinewave_harmonics
gen_sinewave_am_tone
gen_sinewave_am_noise
gen_sinewave_nyquist_zones
gen_sinewave_multirun
gen_sinewave_static_nonlinearity

generate_jitter_sweep_data

generate_pipeline3s_dout
generate_pipeline8s_dout
generate_sar_dout