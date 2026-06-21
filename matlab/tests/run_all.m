%% Run all unit tests

thisDir = fileparts(mfilename('fullpath'));
matlabRoot = fileparts(thisDir);
addpath(genpath(fullfile(matlabRoot, 'src')));
addpath(genpath(thisDir));
cd(matlabRoot);

run_common

run_aout

run_dout

jitterConfig = fullfile('test_dataset', 'jitter_sweep', 'config.csv');
if exist(jitterConfig, 'file')
    run_jitter_load
else
    fprintf('[run_all] Skipping run_jitter_load; missing %s. Run data_generation/gen_jitter_sweep_data.m first.\n', jitterConfig);
end
