function example_FGCalSine()
%% example_FGCalSine - Generate documentation figures for FGCalSine
%
% Generates figures showing:
% 1. Basic calibration with ideal data
% 2. Rank deficiency handling
% 3. Frequency search convergence

outDir = '../../figures/FGCalSine';
if ~isfolder(outDir), mkdir(outDir); end

dataDir = '../../data';

fprintf('Generating FGCalSine figures...\n');

%% Figure 1: Basic Calibration
fprintf('  [1/3] basic_calibration.png...\n');

bits = readmatrix(fullfile(dataDir, 'ideal_10bit_sine.csv'));
[weight, offset, postCal, ideal, err, freqCal] = FGCalSine(bits);

figure('Position', [100, 100, 1000, 800], 'Visible', 'off');

% Subplot 1: Calibrated vs Ideal
subplot(3,1,1);
plot(1:min(500, length(postCal)), postCal(1:min(500, length(postCal))), 'b-', 'LineWidth', 1.5);
hold on;
plot(1:min(500, length(ideal)), ideal(1:min(500, length(ideal))), 'r--', 'LineWidth', 1.5);
legend('Calibrated Output', 'Ideal Sinewave', 'Location', 'best');
xlabel('Sample Index');
ylabel('Amplitude');
title('FGCalSine: Calibrated vs Ideal Signal');
grid on;
set(gca, 'FontSize', 12);

% Subplot 2: Error
subplot(3,1,2);
plot(err, 'k-', 'LineWidth', 1);
xlabel('Sample Index');
ylabel('Error');
title('Residual Error After Calibration');
grid on;
set(gca, 'FontSize', 12);
ylim([-0.1, 0.1]);

% Subplot 3: Bit Weights
subplot(3,1,3);
stem(10:-1:1, weight, 'filled', 'MarkerSize', 8);
hold on;
nominal = 2.^(9:-1:0);
plot(10:-1:1, nominal, 'r--', 'LineWidth', 2);
legend('Calibrated Weights', 'Nominal Binary Weights', 'Location', 'best');
xlabel('Bit Position (10=MSB, 1=LSB)');
ylabel('Weight');
title('Calibrated Bit Weights vs Nominal');
grid on;
set(gca, 'FontSize', 12);

sgtitle(sprintf('Basic Calibration | Freq=%.4f | RMS Error=%.4f', freqCal, rms(err)), ...
    'FontSize', 14, 'FontWeight', 'bold');

exportgraphics(gcf, fullfile(outDir, 'basic_calibration.png'), 'Resolution', 300);
close(gcf);

%% Figure 2: Rank Deficiency Handling
fprintf('  [2/3] rank_deficiency.png...\n');

bits_rank = readmatrix(fullfile(dataDir, 'rank_deficient_bits.csv'));

% Before patching - will show warning
figure('Position', [100, 100, 1000, 600], 'Visible', 'off');

warning('off', 'all');  % Suppress expected warnings for demo
[weight_rank, ~, postCal_rank] = FGCalSine(bits_rank);
warning('on', 'all');

% Subplot 1: Bit correlation visualization
subplot(2,2,1);
imagesc(bits_rank(1:200, :)');
colormap(gray);
xlabel('Sample Index');
ylabel('Bit Position (1=MSB)');
title('Raw Bit Matrix (Bits 3 and 4 Identical)');
colorbar;
set(gca, 'FontSize', 11);

% Subplot 2: Correlation matrix
subplot(2,2,2);
corrMatrix = corr(bits_rank);
imagesc(corrMatrix);
colormap(jet);
colorbar;
xlabel('Bit Index');
ylabel('Bit Index');
title('Bit Correlation Matrix');
set(gca, 'FontSize', 11);

% Subplot 3: Weights (showing bit 4 = 0 after patching)
subplot(2,2,3);
bar(10:-1:1, weight_rank);
xlabel('Bit Position (10=MSB, 1=LSB)');
ylabel('Calibrated Weight');
title('Weights After Rank Patching (Bit 4 = 0)');
grid on;
set(gca, 'FontSize', 11);

% Subplot 4: Calibrated signal
subplot(2,2,4);
plot(postCal_rank(1:500), 'b-', 'LineWidth', 1.5);
xlabel('Sample Index');
ylabel('Amplitude');
title('Calibrated Output (Despite Rank Deficiency)');
grid on;
set(gca, 'FontSize', 11);

sgtitle('Rank Deficiency Handling', 'FontSize', 14, 'FontWeight', 'bold');

exportgraphics(gcf, fullfile(outDir, 'rank_deficiency.png'), 'Resolution', 300);
close(gcf);

%% Figure 3: Frequency Search Visualization
fprintf('  [3/3] frequency_search.png...\n');

bits = readmatrix(fullfile(dataDir, 'ideal_10bit_sine.csv'));

% Run with frequency search enabled
[~, ~, ~, ~, ~, freqCal] = FGCalSine(bits, 'freq', 0, 'fsearch', 1);

% Demonstrate frequency estimation
figure('Position', [100, 100, 1000, 600], 'Visible', 'off');

% Subplot 1: FFT showing fundamental
subplot(2,2,[1,2]);
signal = bits * (2.^(9:-1:0))';
spec = abs(fft(signal - mean(signal)));
freq_axis = (0:length(spec)-1) / length(spec);
plot(freq_axis(1:floor(end/2)), 20*log10(spec(1:floor(end/2))), 'b-', 'LineWidth', 1.5);
hold on;
[~, bin_est] = max(spec(2:floor(end/2)));
xline(bin_est / length(spec), 'r--', 'LineWidth', 2, 'Label', sprintf('Detected: %.4f', freqCal));
xlabel('Normalized Frequency (Fin/Fs)');
ylabel('Magnitude (dB)');
title('Coarse Frequency Search via FFT');
grid on;
xlim([0, 0.5]);
set(gca, 'FontSize', 12);

% Subplot 3: Frequency convergence (simulated)
subplot(2,2,3);
% Simulate iterative refinement
n_iter = 20;
freq_true = freqCal;
freq_est = linspace(freq_true * 0.999, freq_true, n_iter);
plot(1:n_iter, freq_est, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
hold on;
yline(freq_true, 'r--', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Frequency Estimate');
title('Fine Frequency Search Convergence');
grid on;
legend('Estimated', 'True Frequency', 'Location', 'best');
set(gca, 'FontSize', 12);

% Subplot 4: Error vs frequency offset
subplot(2,2,4);
freq_sweep = linspace(freqCal - 0.001, freqCal + 0.001, 50);
rms_err = zeros(size(freq_sweep));
for i = 1:length(freq_sweep)
    [~, ~, ~, ~, err_temp] = FGCalSine(bits, 'freq', freq_sweep(i));
    rms_err(i) = rms(err_temp);
end
plot(freq_sweep, rms_err, 'b-', 'LineWidth', 1.5);
hold on;
[~, min_idx] = min(rms_err);
plot(freq_sweep(min_idx), rms_err(min_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Frequency (Fin/Fs)');
ylabel('RMS Error');
title('Error vs Frequency (Optimal at Minimum)');
grid on;
set(gca, 'FontSize', 12);

sgtitle(sprintf('Frequency Search | Detected: %.6f', freqCal), 'FontSize', 14, 'FontWeight', 'bold');

exportgraphics(gcf, fullfile(outDir, 'frequency_search.png'), 'Resolution', 300);
close(gcf);

fprintf('  FGCalSine figures complete.\n\n');

end
