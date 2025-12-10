%% Centralized Configuration for Aout Test
common_test_aout;

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);
    [~, datasetName, ~] = fileparts(currentFilename);

    read_data = readmatrix(dataFilePath);

    % Extract static nonlinearity coefficients (updated to match Python)
    order = 3;
    [k2, k3, fitted_sine, fitted_transfer] = fitstaticnl(read_data, order);

    fprintf('  [Static non-linearity: k2=%.6f, k3=%.6f]\n', k2, k3);

    % Prepare plotting data (match Python example)
    % 1. Measured residual: deviation from the fundamental sine wave
    residual = read_data - fitted_sine;

    % 2. Fitted curve: use the smooth transfer curve directly
    nonlinearity_curve = fitted_transfer.y - fitted_transfer.x;

    % Create visualization of nonlinearity error
    figure('Position', [100, 100, 800, 600], "Visible", verbose);

    % Plot measured residual and fitted model
    plot(fitted_sine, residual, 'b.', 'MarkerSize', 1, 'DisplayName', 'Measured');
    hold on;
    plot(fitted_transfer.x, nonlinearity_curve, 'r-', 'LineWidth', 2, 'DisplayName', 'Fitted Model');
    hold off;
    grid on;
    xlabel('Input Amplitude (V)');
    ylabel('Nonlinearity Error (V)');
    title(sprintf('Static Nonlinearity: k2=%.6f, k3=%.6f', k2, k3));
    legend('Location', 'best');

    % Save outputs
    subFolder = fullfile(outputDir, datasetName, mfilename);
    figureName = sprintf("%s_%s_matlab.png", mfilename, datasetName);
    saveFig(figureDir, figureName, verbose);

    % Save all return values for comparison with Python
    saveVariable(subFolder, k2, verbose);
    saveVariable(subFolder, k3, verbose);
    saveVariable(subFolder, fitted_sine, verbose);
    % Save fitted_transfer struct fields separately (create temporary variables)
    fitted_transfer_x = fitted_transfer.x;
    fitted_transfer_y = fitted_transfer.y;
    saveVariable(subFolder, fitted_transfer_x, verbose);
    saveVariable(subFolder, fitted_transfer_y, verbose);
end
