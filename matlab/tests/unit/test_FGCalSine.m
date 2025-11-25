%% test_FGCalSine.m - Unit test for FGCalSine function
% Tests the FGCalSine foreground calibration function with SAR/Pipeline data
%
% Output structure:
%   test_output/<data_set_name>/test_FGCalSine/
%       weight_matlab.csv       - calibrated bit weights
%       offset_matlab.csv       - DC offset
%       freqCal_matlab.csv      - calibrated frequency
%       postCal_matlab.csv      - first 1000 samples of calibrated output
%       ideal_matlab.csv        - first 1000 samples of ideal sinewave
%       err_matlab.csv          - first 1000 samples of residual error

close all; clc; clear;

%% Configuration
inputDir = "dataset";
outputDir = "test_output";

% Test datasets - leave empty to auto-search
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'dout_*.csv');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% Test Loop
fprintf('=== test_FGCalSine.m ===\n');
fprintf('[Testing] %d datasets...\n\n', length(filesList));

for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);

    if ~isfile(dataFilePath)
        fprintf('[%d/%d] %s - NOT FOUND, skipping\n\n', k, length(filesList), currentFilename);
        continue;
    end
    fprintf('[%d/%d] [Processing] %s\n', k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    % Extract dataset name
    [~, datasetName, ~] = fileparts(currentFilename);

    % Create output subfolder
    subFolder = fullfile(outputDir, datasetName, 'test_FGCalSine');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    %% Run FGCalSine
    [weight, offset, postCal, ideal, err, freqCal] = FGCalSine(read_data, 'freq', 0, 'order', 5);

    % Save weight to CSV
    weightTable = table((1:length(weight))', weight', ...
        'VariableNames', {'bit_index', 'weight'});
    weightPath = fullfile(subFolder, 'weight_matlab.csv');
    writetable(weightTable, weightPath);
    fprintf('  [Saved] %s\n', weightPath);

    % Save offset to CSV
    offsetTable = table(offset, 'VariableNames', {'offset'});
    offsetPath = fullfile(subFolder, 'offset_matlab.csv');
    writetable(offsetTable, offsetPath);
    fprintf('  [Saved] %s\n', offsetPath);

    % Save freqCal to CSV
    freqCalTable = table(freqCal, 'VariableNames', {'freqCal'});
    freqCalPath = fullfile(subFolder, 'freqCal_matlab.csv');
    writetable(freqCalTable, freqCalPath);
    fprintf('  [Saved] %s\n', freqCalPath);

    % Save postCal, ideal, err (first 1000 samples)
    N_save = min(1000, length(postCal));

    postCalTable = table(postCal(1:N_save)', 'VariableNames', {'postCal'});
    postCalPath = fullfile(subFolder, 'postCal_matlab.csv');
    writetable(postCalTable, postCalPath);
    fprintf('  [Saved] %s\n', postCalPath);

    idealTable = table(ideal(1:N_save)', 'VariableNames', {'ideal'});
    idealPath = fullfile(subFolder, 'ideal_matlab.csv');
    writetable(idealTable, idealPath);
    fprintf('  [Saved] %s\n', idealPath);

    errTable = table(err(1:N_save)', 'VariableNames', {'err'});
    errPath = fullfile(subFolder, 'err_matlab.csv');
    writetable(errTable, errPath);
    fprintf('  [Saved] %s\n', errPath);

    % Print summary
    fprintf('  [Results] freqCal=%.8f, offset=%.6f, weight_sum=%.6f, err_rms=%.6f\n\n', ...
        freqCal, offset, sum(weight), rms(err));
end

fprintf('[test_FGCalSine COMPLETE]\n');
