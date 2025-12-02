%% Centralized Configuration for Aout Test
common_test_aout;

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [enob, sndr, sfdr, snr, thd, sigpwr, noi, nsd, h] = plotspec(read_data, 'label', 1, 'harmonic', 5, 'OSR', 1);
    set(gca, "FontSize",16)

    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, mfilename);


    figureName = sprintf("%s_%s_matlab.png", mfilename, datasetName);
    saveFig(figureDir, figureName, verbose);
    saveVariable(subFolder, enob, verbose);
    saveVariable(subFolder, sndr, verbose);
    saveVariable(subFolder, sfdr, verbose);
    saveVariable(subFolder, snr, verbose);
    saveVariable(subFolder, thd, verbose);
    saveVariable(subFolder, sigpwr, verbose);
    saveVariable(subFolder, noi, verbose);
    saveVariable(subFolder, nsd, verbose);
end
