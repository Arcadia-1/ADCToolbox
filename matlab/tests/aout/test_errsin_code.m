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

    % Error histogram analysis
    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [emean_code, erms_code, code_axis] = ...
        errsin(read_data, 'bin', 256, 'fin', freq, 'disp', 1, 'xaxis', 'value');
    sgtitle("Error - Code");
    set(findall(gcf, 'Type', 'axes'), 'FontSize', 14);

    figureName = sprintf("%s_%s_matlab.png", mfilename, datasetName);
    saveFig(figureDir, figureName, verbose);

    subFolder = fullfile(outputDir, datasetName, mfilename);
    saveVariable(subFolder, code_axis, verbose);
    saveVariable(subFolder, emean_code, verbose);
    saveVariable(subFolder, erms_code, verbose);
end
