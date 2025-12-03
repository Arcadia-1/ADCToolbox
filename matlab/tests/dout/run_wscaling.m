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
    [weight_cal, ~, ~, ~, ~, ~] = wcalsine(read_data, 'freq', 0, 'order', 5);
    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    radix = weightScaling(weight_cal);

    % Save outputs
    subFolder = fullfile(outputDir, datasetName, mfilename);
    figureName = sprintf("%s_%s_matlab.png", mfilename, datasetName);
    saveFig(figureDir, figureName, verbose);
    saveVariable(subFolder, radix, verbose);
    saveVariable(subFolder, weight_cal, verbose);
end
