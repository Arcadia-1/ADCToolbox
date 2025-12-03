%% Centralized Configuration for Dout Test
common_test_dout;

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);
    [~, datasetName, ~] = fileparts(currentFilename);

    % Run wcalsine to get calibrated weights
    [weights_cal, ~, ~, ~, ~, ~] = wcalsine(read_data);

    % Run ovfchk and get overflow statistics
    [range_min, range_max, ovf_percent_zero, ovf_percent_one] = ovfchk(read_data, weights_cal);

    % Create visualization plot
    figure('Position', [100, 100, 1000, 600], "Visible", verbose);
    ovfchk(read_data, weights_cal);  % Call without outputs to generate plot
    title('Overflow Check');

    % Save outputs
    subFolder = fullfile(outputDir, datasetName, mfilename);
    figureName = sprintf("%s_%s_matlab.png", mfilename, datasetName);
    saveFig(figureDir, figureName, verbose);

    % Save variables
    saveVariable(subFolder, range_min, verbose);
    saveVariable(subFolder, range_max, verbose);
    saveVariable(subFolder, ovf_percent_zero, verbose);
    saveVariable(subFolder, ovf_percent_one, verbose);
end
