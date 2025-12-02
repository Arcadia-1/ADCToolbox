%% Centralized Configuration for Dout Test
common_test_dout;

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);
    [~, datasetName, ~] = fileparts(currentFilename);
    [N, M] = size(read_data);

    % Calculate nominal binary weights
    nomWeight = 2.^(M-1:-1:0)';

    % Pre-calibration: Convert using nominal weights
    preCal = read_data * nomWeight;

    % Run wcalsine (foreground calibration)
    [weight, offset, postCal, ideal, err, freqCal] = wcalsine(read_data, 'freq', 0, 'order', 5);

    % Plot spectrum before calibration
    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [ENoB_before, SNDR_before, SFDR_before, SNR_before, THD_before, pwr_before, NF_before, ~, ~] = plotspec(preCal, 'label', 1, 'harmonic', 5, 'OSR', 1, 'NFMethod', 0);
    title('Spectrum - Before Calibration');

    figureName_before = sprintf("%s_%s_before_matlab.png", datasetName, mfilename);
    saveFig(figureDir, figureName_before, verbose);

    % Plot spectrum after calibration
    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [ENoB_after, SNDR_after, SFDR_after, SNR_after, THD_after, pwr_after, NF_after, ~, ~] = plotspec(postCal, 'label', 1, 'harmonic', 5, 'OSR', 1, 'NFMethod', 0);
    title('Spectrum - After Calibration');

    figureName_after = sprintf("%s_%s_after_matlab.png", datasetName, mfilename);
    saveFig(figureDir, figureName_after, verbose);

    % Save outputs
    subFolder = fullfile(outputDir, datasetName, mfilename);
    saveVariable(subFolder, weight, verbose);
    saveVariable(subFolder, offset, verbose);
    saveVariable(subFolder, postCal, verbose);
    saveVariable(subFolder, ideal, verbose);
    saveVariable(subFolder, err, verbose);
    saveVariable(subFolder, freqCal, verbose);
    saveVariable(subFolder, preCal, verbose);
    saveVariable(subFolder, ENoB_before, verbose);
    saveVariable(subFolder, ENoB_after, verbose);
end
