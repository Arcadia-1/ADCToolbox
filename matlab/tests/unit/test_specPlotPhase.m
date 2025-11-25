%% test_specPlotPhase.m - Unit test for specPlotPhase function
% Tests the specPlotPhase function with various sinewave datasets
%
% Output structure:
%   test_output/<data_set_name>/test_specPlotPhase/
%       phase_matlab.png        - Phase polar plot
%       phase_data_matlab.csv   - Spectrum data with phase information

close all; clc; clear;

%% Configuration
inputDir = "dataset";
outputDir = "test_output";

% Test datasets - leave empty to auto-search
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv', 'batch_sinewave_*.csv');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% Test Loop
fprintf('=== test_specPlotPhase.m ===\n');
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
    subFolder = fullfile(outputDir, datasetName, 'test_specPlotPhase');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    %% Run specPlotPhase
    figure('Visible', 'off');
    [h, spec, phi, bin] = specPlotPhase(read_data, 'harmonic', 10);
    title(['specPlotPhase: ', titleString]);

    % Save plot
    plotPath = fullfile(subFolder, 'phase_matlab.png');
    saveas(gcf, plotPath);
    fprintf('  [Saved] %s\n', plotPath);

    % Save data to CSV
    csvData = [real(spec'), imag(spec'), abs(spec'), angle(spec'), real(phi'), imag(phi')];
    csvData = csvData(2:end,:);
    csvPath = fullfile(subFolder, 'phase_data_matlab.csv');
    csvHeader = 'spec_real,spec_imag,spec_mag,spec_phase,phi_real,phi_imag';
    writematrix(csvData, csvPath);

    % Add header manually by prepending to file
    fid = fopen(csvPath, 'r');
    fileContent = fread(fid, '*char')';
    fclose(fid);
    fid = fopen(csvPath, 'w');
    fprintf(fid, '%s\n%s', csvHeader, fileContent);
    fclose(fid);

    fprintf('  [Saved] %s\n', csvPath);
    fprintf('  [Results] fundamental bin: %d\n\n', bin);
    close(gcf);
end

fprintf('[test_specPlotPhase COMPLETE]\n');
