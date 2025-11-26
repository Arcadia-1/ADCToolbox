%% test_errHistSine_polyfit.m - Test polynomial regression for static nonlinearity
% Tests the errHistSine function with polynomial fitting in code mode
%
% Output structure:
%   test_output/<data_set_name>/test_errHistSine_polyfit/
%       polycoeff_matlab.csv            - polynomial coefficients
%       code_histogram_polyfit_matlab.csv - code, emean, erms
%       errHistSine_polyfit_matlab.png  - plot with polynomial fit

close all; clc; clear; warning("off")

%% Configuration
inputDir = "dataset/non_lin";  % Subfolder for nonlinearity test data
outputDir = "test_output";

% Polynomial order to test
polyOrders = [3];  % Use 3 to capture HD2 (k2) and HD3 (k3)
% polyOrders = [3, 5, 7];  % Uncomment to compare multiple orders

% Test datasets - leave empty to auto-search
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_HD*.csv');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% Test Loop
fprintf('=== test_errHistSine_polyfit.m ===\n');
fprintf('[Testing] %d datasets with polynomial orders: %s\n\n', ...
    length(filesList), mat2str(polyOrders));

for k = 1 %:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);

    if ~isfile(dataFilePath)
        fprintf('[%d/%d] %s - NOT FOUND, skipping\n\n', k, length(filesList), currentFilename);
        continue;
    end
    fprintf('[%d/%d] [Processing] %s\n', k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    % Extract dataset name
    [~, datasetName, ~] = fileparts(currentFilename);
    titleString = replace(datasetName, '_', '\_');

    % Create output subfolder
    subFolder = fullfile(outputDir, datasetName, 'test_errHistSine_polyfit');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    % Get frequency
    [~, freq, ~, ~, ~] = sineFit(read_data);

    % Extract coefficients using FFT method (most accurate)
    [k1_fft, k2_fft, k3_fft, k4_fft, k5_fft] = extractCoeffsFromFFT(read_data, freq);
    fprintf('  [FFT-based Extraction (Reference)]\n');
    fprintf('    k1 (gain) = %.6e\n', k1_fft);
    fprintf('    k2 (HD2)  = %.6e\n', k2_fft);
    fprintf('    k3 (HD3)  = %.6e\n', k3_fft);

    % Calculate expected coefficients from filename
    [k0_exp, k1_exp, k2_exp, k3_exp, k4_exp, k5_exp] = calcExpectedCoeffs(currentFilename);
    fprintf('  [Expected Coefficients from Generation]\n');
    fprintf('    k0 (offset) = %.6e\n', k0_exp);
    fprintf('    k1 (gain)   = %.6e\n', k1_exp);
    if ~isnan(k2_exp), fprintf('    k2 (HD2)    = %.6e\n', k2_exp); end
    if ~isnan(k3_exp), fprintf('    k3 (HD3)    = %.6e\n', k3_exp); end
    if ~isnan(k4_exp), fprintf('    k4 (HD4)    = %.6e\n', k4_exp); end
    if ~isnan(k5_exp), fprintf('    k5 (HD5)    = %.6e\n', k5_exp); end

    % Save expected coefficients to CSV
    expCoeffs = [k0_exp, k1_exp];
    expNames = {'k0_offset_expected', 'k1_gain_expected'};
    if ~isnan(k2_exp)
        expCoeffs = [expCoeffs, k2_exp];
        expNames = [expNames, {'k2_nonlin_expected'}];
    end
    if ~isnan(k3_exp)
        expCoeffs = [expCoeffs, k3_exp];
        expNames = [expNames, {'k3_nonlin_expected'}];
    end
    if ~isnan(k4_exp)
        expCoeffs = [expCoeffs, k4_exp];
        expNames = [expNames, {'k4_nonlin_expected'}];
    end
    if ~isnan(k5_exp)
        expCoeffs = [expCoeffs, k5_exp];
        expNames = [expNames, {'k5_nonlin_expected'}];
    end
    expTable = array2table(expCoeffs, 'VariableNames', expNames);
    expPath = fullfile(subFolder, 'expected_coefficients_matlab.csv');
    writetable(expTable, expPath);
    fprintf('  [Saved] %s\n', expPath);

    % Save FFT-extracted coefficients to CSV
    fftCoeffs = [k1_fft, k2_fft, k3_fft];
    fftNames = {'k1_gain_fft', 'k2_HD2_fft', 'k3_HD3_fft'};
    fftTable = array2table(fftCoeffs, 'VariableNames', fftNames);
    fftPath = fullfile(subFolder, 'coefficients_fft_matlab.csv');
    writetable(fftTable, fftPath);
    fprintf('  [Saved] %s\n', fftPath);

    %% Test different polynomial orders
    for p = 1:length(polyOrders)
        polyOrder = polyOrders(p);
        fprintf('  [Testing] Polynomial order %d\n', polyOrder);

        % Run errHistSine with polynomial fitting
        figure('Position', [100, 100, 800, 600], 'Visible', 'on');
        [emean_code, erms_code, code_axis, ~, ~, ~, ~, polycoeff] = ...
            errHistSine(read_data, 'bin', 256, 'fin', freq, 'disp', 1, ...
                       'mode', 1, 'polyorder', polyOrder);
        sgtitle(['errHistSine Poly Fit (order=', num2str(polyOrder), '): ', titleString]);

        % Save plot
        plotPath = fullfile(subFolder, sprintf('errHistSine_polyfit_order%d_matlab.png', polyOrder));
        saveas(gcf, plotPath);
        fprintf('    [Saved] %s\n', plotPath);
        % close(gcf);

        % Save polynomial coefficients
        if ~isempty(polycoeff)
            % Ensure polycoeff is a row vector for table creation
            polycoeff_row = polycoeff(:)';  % Convert to row vector
            polyTable = array2table(polycoeff_row, ...
                'VariableNames', arrayfun(@(n) sprintf('p%d', n), polyOrder:-1:0, 'UniformOutput', false));
            polyPath = fullfile(subFolder, sprintf('polycoeff_order%d_matlab.csv', polyOrder));
            writetable(polyTable, polyPath);
            fprintf('    [Saved] %s\n', polyPath);

            % Display coefficients (INL polynomial fit)
            fprintf('    [INL Polynomial Coefficients] ');
            for i = 1:length(polycoeff)
                fprintf('p%d=%.6e ', polyOrder-i+1, polycoeff(i));
            end
            fprintf('\n');

            % Extract nonlinearity coefficients (k2, k3, ...)
            % Transfer function: y = k0 + k1*x + k2*x^2 + k3*x^3 + ...
            % INL = y - x, so y = x + INL(x)
            % If INL(x_norm) = p_n*x^n + ... + p_1*x + p_0
            % Then the transfer function is: y = p_0 + (1+p_1)*x + p_2*x^2 + p_3*x^3 + ...
            fprintf('    [Transfer Function Coefficients (normalized x ∈ [-1,1])]\n');
            fprintf('      k0 (offset) = %.6e\n', polycoeff(end));
            if length(polycoeff) >= 2
                fprintf('      k1 (gain)   = %.6e  [ideal=1.0, actual≈%.6f]\n', ...
                    1 + polycoeff(end-1), 1 + polycoeff(end-1));
            end
            if length(polycoeff) >= 3
                fprintf('      k2 (2nd order nonlinearity) = %.6e\n', polycoeff(end-2));
            end
            if length(polycoeff) >= 4
                fprintf('      k3 (3rd order nonlinearity) = %.6e\n', polycoeff(end-3));
            end
            for i = 4:length(polycoeff)-1
                fprintf('      k%d (%dth order) = %.6e\n', i, i, polycoeff(end-i));
            end

            % Compare extracted vs expected coefficients
            fprintf('    [Comparison: Polynomial Fit vs FFT vs Expected]\n');
            k0_extracted = polycoeff(end);
            k1_extracted = 1 + polycoeff(end-1);

            fprintf('      k1 (gain):\n');
            fprintf('        Polyfit:  %.6e  (error vs expected: %.2f%%, vs FFT: %.2f%%)\n', ...
                k1_extracted, abs(k1_extracted - k1_exp) / abs(k1_exp) * 100, ...
                abs(k1_extracted - k1_fft) / abs(k1_fft) * 100);
            fprintf('        FFT:      %.6e  (error vs expected: %.2f%%)\n', ...
                k1_fft, abs(k1_fft - k1_exp) / abs(k1_exp) * 100);
            fprintf('        Expected: %.6e\n', k1_exp);

            if length(polycoeff) >= 3 && ~isnan(k2_exp)
                k2_extracted = polycoeff(end-2);
                fprintf('      k2 (HD2):\n');
                fprintf('        Polyfit:  %.6e  (error vs expected: %.2f%%, vs FFT: %.2f%%)\n', ...
                    k2_extracted, abs(k2_extracted - k2_exp) / abs(k2_exp) * 100, ...
                    abs(k2_extracted - k2_fft) / abs(k2_fft) * 100);
                fprintf('        FFT:      %.6e  (error vs expected: %.2f%%)\n', ...
                    k2_fft, abs(k2_fft - k2_exp) / abs(k2_exp) * 100);
                fprintf('        Expected: %.6e\n', k2_exp);
            end

            if length(polycoeff) >= 4 && ~isnan(k3_exp)
                k3_extracted = polycoeff(end-3);
                fprintf('      k3 (HD3):\n');
                fprintf('        Polyfit:  %.6e  (error vs expected: %.2f%%, vs FFT: %.2f%%)\n', ...
                    k3_extracted, abs(k3_extracted - k3_exp) / abs(k3_exp) * 100, ...
                    abs(k3_extracted - k3_fft) / abs(k3_fft) * 100);
                fprintf('        FFT:      %.6e  (error vs expected: %.2f%%)\n', ...
                    k3_fft, abs(k3_fft - k3_exp) / abs(k3_exp) * 100);
                fprintf('        Expected: %.6e\n', k3_exp);
            end

            % Save comparison table (expected vs polyfit vs FFT)
            compNames = {};
            compExpected = [];
            compPolyfit = [];
            compFFT = [];
            compPolyfitError = [];
            compFFTError = [];

            % k1
            compNames = [compNames, {'k1_gain'}];
            compExpected = [compExpected, k1_exp];
            compPolyfit = [compPolyfit, k1_extracted];
            compFFT = [compFFT, k1_fft];
            compPolyfitError = [compPolyfitError, abs(k1_extracted - k1_exp) / abs(k1_exp) * 100];
            compFFTError = [compFFTError, abs(k1_fft - k1_exp) / abs(k1_exp) * 100];

            % k2
            if length(polycoeff) >= 3 && ~isnan(k2_exp)
                k2_extracted = polycoeff(end-2);
                compNames = [compNames, {'k2_HD2'}];
                compExpected = [compExpected, k2_exp];
                compPolyfit = [compPolyfit, k2_extracted];
                compFFT = [compFFT, k2_fft];
                compPolyfitError = [compPolyfitError, abs(k2_extracted - k2_exp) / abs(k2_exp) * 100];
                compFFTError = [compFFTError, abs(k2_fft - k2_exp) / abs(k2_exp) * 100];
            end

            % k3
            if length(polycoeff) >= 4 && ~isnan(k3_exp)
                k3_extracted = polycoeff(end-3);
                compNames = [compNames, {'k3_HD3'}];
                compExpected = [compExpected, k3_exp];
                compPolyfit = [compPolyfit, k3_extracted];
                compFFT = [compFFT, k3_fft];
                compPolyfitError = [compPolyfitError, abs(k3_extracted - k3_exp) / abs(k3_exp) * 100];
                compFFTError = [compFFTError, abs(k3_fft - k3_exp) / abs(k3_exp) * 100];
            end

            compTable = table(compNames', compExpected', compPolyfit', compPolyfitError', ...
                              compFFT', compFFTError', ...
                'VariableNames', {'Coefficient', 'Expected', 'Polyfit', 'Polyfit_Error_Pct', ...
                                  'FFT', 'FFT_Error_Pct'});
            compPath = fullfile(subFolder, sprintf('comparison_order%d_matlab.csv', polyOrder));
            writetable(compTable, compPath);
            fprintf('    [Saved] %s\n', compPath);

            % Save transfer function coefficients to CSV
            k0 = polycoeff(end);
            k1 = 1 + polycoeff(end-1);
            tfCoeffs = [k0, k1];
            tfNames = {'k0_offset', 'k1_gain'};
            for i = 2:length(polycoeff)-1
                tfCoeffs = [tfCoeffs, polycoeff(end-i)];
                tfNames = [tfNames, {sprintf('k%d_nonlin', i)}];
            end
            tfTable = array2table(tfCoeffs, 'VariableNames', tfNames);
            tfPath = fullfile(subFolder, sprintf('transfer_function_coeffs_order%d_matlab.csv', polyOrder));
            writetable(tfTable, tfPath);
            fprintf('    [Saved] %s\n', tfPath);

            % Calculate fit quality (R-squared)
            valid_idx = ~isnan(emean_code);
            if sum(valid_idx) > 0
                dat_min = min(code_axis);
                dat_max = max(code_axis);
                x_norm = 2 * (code_axis(valid_idx) - dat_min) / (dat_max - dat_min) - 1;
                y_fit = polyval(polycoeff, x_norm);
                y_actual = emean_code(valid_idx);

                SS_res = sum((y_actual - y_fit).^2);
                SS_tot = sum((y_actual - mean(y_actual)).^2);
                R_squared = 1 - SS_res / SS_tot;
                fprintf('    [R-squared] %.6f\n', R_squared);
            end
        else
            fprintf('    [Warning] Polynomial fitting failed\n');
        end

        % Save histogram data with polynomial order info
        codeHistTable = table(code_axis', emean_code', erms_code', ...
            'VariableNames', {'code', 'emean', 'erms'});
        codeHistPath = fullfile(subFolder, sprintf('code_histogram_polyfit_order%d_matlab.csv', polyOrder));
        writetable(codeHistTable, codeHistPath);
        fprintf('    [Saved] %s\n', codeHistPath);
    end

    fprintf('\n');
end

fprintf('[test_errHistSine_polyfit COMPLETE]\n');
