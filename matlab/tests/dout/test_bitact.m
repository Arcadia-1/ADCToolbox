%% Centralized Configuration for Dout Test
common_test_dout;

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);
    [~, datasetName, ~] = fileparts(currentFilename);

    % Run bitActivity
    figure('Position', [100, 100, 1000, 750], "Visible", verbose);
    bit_usage = bitact(read_data, 'annotateExtremes', true);
    title("Bit activity");

    % Save outputs
    subFolder = fullfile(outputDir, datasetName, mfilename);
    figureName = sprintf("%s_%s_matlab.png", datasetName, mfilename);
    saveFig(figureDir, figureName, verbose);
    saveVariable(subFolder, bit_usage, verbose);
end
