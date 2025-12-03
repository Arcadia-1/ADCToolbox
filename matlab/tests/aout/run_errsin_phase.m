%% Centralized Configuration for Aout Test
common_test_aout;

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);
    [~, datasetName, ~] = fileparts(currentFilename);

    read_data = readmatrix(dataFilePath);
    [~, freq, ~, ~, ~] = sinfit(read_data);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [emean, erms, xx, anoi, pnoi] = errsin(read_data, 'bin', 360, 'fin', freq, 'disp', 1, 'xaxis', 'phase');
    sgtitle("Error - Phase");
    set(findall(gcf, 'Type', 'axes'), 'FontSize', 14);

    figureName = sprintf("%s_%s_matlab.png", mfilename, datasetName);
    saveFig(figureDir, figureName, verbose);

    subFolder = fullfile(outputDir, datasetName, mfilename);
    saveVariable(subFolder, anoi, verbose);
    saveVariable(subFolder, pnoi, verbose);
    saveVariable(subFolder, xx, verbose);
    saveVariable(subFolder, emean, verbose);
    saveVariable(subFolder, erms, verbose);
end
