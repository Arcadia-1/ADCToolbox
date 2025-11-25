%% test_FGCalSine_overflowChk.m - Unit test for overflowChk function
% Tests the overflowChk function with SAR ADC digital output data
%
% Output structure:
%   test_output/<data_set_name>/test_overflowChk/
%       overflowChk_matlab.png  - overflow check plot
%       data_decom_matlab.csv   - decomposed residue data (N samples x M bits)

close all; clc; clear;
warning("off");

%% Configuration
inputDir = "dataset";
outputDir = "test_output";

% Test datasets - leave empty to auto-search
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'dout_SAR_*.csv');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% Test Loop
fprintf('=== test_FGCalSine_overflowChk.m ===\n');
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
    titleString = replace(datasetName, '_', '\_');

    % Create output subfolder
    subFolder = fullfile(outputDir, datasetName, 'test_overflowChk');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    %% Run FGCalSine to get calibrated weights
    weights_cal = FGCalSine(read_data);

    %% Run overflowChk
    figure('Visible', 'off');
    data_decom = overflowChk(read_data, weights_cal);
    title(titleString);

    % Save plot
    plotPath = fullfile(subFolder, 'overflowChk_matlab.png');
    saveas(gcf, plotPath);
    fprintf('  [Saved] %s\n', plotPath);
    close(gcf);

    % Save data_decom to CSV
    [N, M] = size(data_decom);
    % Create column names: bit_M, bit_M-1, ..., bit_1, bit_0
    varNames = arrayfun(@(x) sprintf('bit_%d', x), M-1:-1:0, 'UniformOutput', false);
    dataDecomTable = array2table(data_decom, 'VariableNames', varNames);
    dataDecomPath = fullfile(subFolder, 'data_decom_matlab.csv');
    writetable(dataDecomTable, dataDecomPath);
    fprintf('  [Saved] %s\n\n', dataDecomPath);
end

fprintf('[test_FGCalSine_overflowChk COMPLETE]\n');
