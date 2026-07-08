%% Centralized Configuration for Aout Test
common_test_aout;

%% Test Loop - LMS Mode
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, mfilename);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [~, harm_phase, harm_mag, freq, noise_dB] = plotphase(read_data, 'harmonic', 5, 'mode', 'LMS');
    set(gca, "FontSize",16)

    figureName = sprintf("%s_%s_matlab.png", mfilename, datasetName);
    saveFig(figureDir, figureName, verbose);
    saveVariable(subFolder, harm_phase, verbose);
    saveVariable(subFolder, harm_mag, verbose);
    saveVariable(subFolder, freq, verbose);
    saveVariable(subFolder, noise_dB, verbose);
end
