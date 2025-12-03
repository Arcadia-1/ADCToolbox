%% Centralized Configuration for Aout Test
common_test_aout;

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);
    [~, datasetName, ~] = fileparts(currentFilename);

    read_data = readmatrix(dataFilePath);

    % Get ideal fit for plotting
    [sig_fit, freq, mag, dc, phi] = sinfit(read_data);

    % Extract static nonlinearity coefficients
    order = 3;
    [k1, k2, k3, polycoeff, fit_curve] = fitstaticnl(read_data, order, freq);

    fprintf('  [Static non-linearity: k1=%.6f, k2=%.6f, k3=%.6f]\n', k1, k2, k3);

    % Create visualization of transfer function
    figure('Position', [100, 100, 800, 600], "Visible", verbose);

    x_ideal = sig_fit - dc;
    y_actual = read_data - mean(read_data);
    y_fit = fit_curve - dc;

    residual = y_actual - y_fit;
    plot(x_ideal, residual, 'b.', 'MarkerSize', 3);
    hold on;
    yline(0, 'r--', 'LineWidth', 1.5);
    hold off;
    grid on;
    xlabel('Ideal Input (zero-mean)');
    ylabel('Residual Error');
    title(sprintf('Fit Residual: k1=%.6f, k2=%.6f, k3=%.6f (RMS=%.2e)', k1, k2, k3, rms(residual)));


    % Save outputs
    subFolder = fullfile(outputDir, datasetName, mfilename);
    figureName = sprintf("%s_%s_matlab.png", mfilename, datasetName);
    saveFig(figureDir, figureName, verbose);

    saveVariable(subFolder, k1, verbose);
    saveVariable(subFolder, k2, verbose);
    saveVariable(subFolder, k3, verbose);
    saveVariable(subFolder, polycoeff, verbose);
    saveVariable(subFolder, fit_curve, verbose);
    saveVariable(subFolder, residual, verbose);
end
