%% Centralized Configuration for Aout Test
common_test_aout;

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);
    [~, datasetName, ~] = fileparts(currentFilename);

    read_data = readmatrix(dataFilePath);
    err_data = geterrsin(read_data);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [acf, lags] = errac(err_data, 'MaxLag', 200, 'Normalize', 0);
    title("Error Auto-Correlation");
    set(gca, "FontSize", 16);

    figureName = sprintf("%s_%s_matlab.png", mfilename, datasetName);
    saveFig(figureDir, figureName, verbose);

    subFolder = fullfile(outputDir, datasetName, mfilename);
    saveVariable(subFolder, lags, verbose);
    saveVariable(subFolder, acf, verbose);
end
