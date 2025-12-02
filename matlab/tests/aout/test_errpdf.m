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
    [~, mu, sigma, KL_divergence, x, fx, gauss_pdf] = errpdf(err_data, ...
        'Resolution', 8, 'FullScale', max(read_data) - min(read_data));
    title("PDF of Error");
    set(gca, "FontSize", 16);

    subFolder = fullfile(outputDir, datasetName, mfilename);

    figureName = sprintf("%s_%s_matlab.png", mfilename, datasetName);
    saveFig(figureDir, figureName, verbose);
    saveVariable(subFolder, mu, verbose);
    saveVariable(subFolder, sigma, verbose);
    saveVariable(subFolder, KL_divergence, verbose);
    saveVariable(subFolder, x, verbose);
    saveVariable(subFolder, fx, verbose);
    saveVariable(subFolder, gauss_pdf, verbose);
end
