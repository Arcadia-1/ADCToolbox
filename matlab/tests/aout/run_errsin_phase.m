%% Centralized Configuration for Aout Test
common_test_aout;

%% Signal Parameters
Fs = 1e9;  % Sampling frequency (1 GHz)
Fin = 1e9; % Input frequency (1 GHz)

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('\n[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);
    [~, datasetName, ~] = fileparts(currentFilename);

    read_data = readmatrix(dataFilePath);
    [~, freq, mag, ~, ~] = sinfit(read_data);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [emean, erms, xx, anoi, pnoi] = errsin(read_data, 'bin', 360, 'fin', freq, 'disp', 1, 'xaxis', 'phase');
    sgtitle("Error - Phase");
    set(findall(gcf, 'Type', 'axes'), 'FontSize', 14);

    figureName = sprintf("%s_%s_matlab.png", mfilename, datasetName);
    saveFig(figureDir, figureName, verbose);

    anoi_uV = anoi * 1e6;
    pnoi_rad  = pnoi;
    pnoi_urad = pnoi * 1e6;

    pnoi_uV = (pnoi * mag) * 1e6;
    jitter_rms = pnoi / (2 * pi * Fin);
    jitter_rms_fs = jitter_rms * 1e15;

    fprintf("[AM_rms=%0.4f uV] [PM_rms=%0.4f urad] -> [%0.4f uV] -> [%0.4f fs]\n", anoi_uV, pnoi_urad, pnoi_uV, jitter_rms_fs)

    subFolder = fullfile(outputDir, datasetName, mfilename);
    saveVariable(subFolder, anoi, verbose);
    saveVariable(subFolder, pnoi, verbose);
    saveVariable(subFolder, jitter_rms, verbose);
    saveVariable(subFolder, xx, verbose);
    saveVariable(subFolder, emean, verbose);
    saveVariable(subFolder, erms, verbose);
end
