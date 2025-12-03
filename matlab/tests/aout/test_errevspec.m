%% Centralized Configuration for Aout Test
common_test_aout;

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);
    [~, datasetName, ~] = fileparts(currentFilename);

    read_data = readmatrix(dataFilePath);
    err_data = read_data - sinfit(read_data);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [ENoB, SNDR, SFDR, SNR, THD, pwr, NF, ~] = errevspec(err_data, 'Fs', 1);
    title('Spectrum of Error Envelope');
    set(gca, "FontSize", 16);

    subFolder = fullfile(outputDir, datasetName, mfilename);

    figureName = sprintf("%s_%s_matlab.png", mfilename, datasetName);
    saveFig(figureDir, figureName, verbose);
    saveVariable(subFolder, ENoB, verbose);
    saveVariable(subFolder, SNDR, verbose);
    saveVariable(subFolder, SFDR, verbose);
    saveVariable(subFolder, SNR, verbose);
    saveVariable(subFolder, THD, verbose);
    saveVariable(subFolder, pwr, verbose);
    saveVariable(subFolder, NF, verbose);
end
