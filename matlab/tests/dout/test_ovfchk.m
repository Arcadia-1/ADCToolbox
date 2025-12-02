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

    % Run ovfchk (overflowChk doesn't return a value, only plots)
    figure('Position', [100, 100, 1000, 600], "Visible", verbose);
    ovfchk(read_data, weights_cal);

    title(sprintf('overflow_chk: %s', datasetName));

    % Save figure
    subFolder = fullfile(outputDir, datasetName, mfilename);
    figureName = sprintf("%s_%s_matlab.png", datasetName, mfilename);
    saveFig(figureDir, figureName, verbose);
end
