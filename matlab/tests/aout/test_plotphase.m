%% Centralized Configuration for Aout Test
common_test_aout;

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    h = plotphase(read_data, 'harmonic', 10, 'mode', 'FFT');
    set(gca, "FontSize",16)

    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, mfilename);

    figureName = sprintf("%s_%s_matlab.png", mfilename, datasetName);
    saveFig(figureDir, figureName, verbose);
end
