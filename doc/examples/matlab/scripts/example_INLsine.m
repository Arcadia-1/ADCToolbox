function example_INLsine()
%% example_INLsine - Generate documentation figures for INLsine
%
% Generates figures showing:
% 1. INL/DNL from ideal ADC (baseline)
% 2. INL/DNL with nonlinearity

outDir = '../../figures/INLsine';
if ~isfolder(outDir), mkdir(outDir); end

dataDir = '../../data';

fprintf('Generating INLsine figures...\n');

%% Figure 1: INL/DNL from data with nonlinearity
fprintf('  [1/2] inl_dnl_analysis.png...\n');

bits_inl = readmatrix(fullfile(dataDir, 'with_inl_dnl.csv'));
[~, ~, postCal] = FGCalSine(bits_inl);

[INL, DNL, code] = INLsine(postCal, 0.01);

figure('Position', [100, 100, 1000, 800], 'Visible', 'off');

% Subplot 1: INL
subplot(3,1,1);
plot(code, INL, 'b-', 'LineWidth', 1.5);
hold on;
yline(0, 'k--', 'LineWidth', 1);
xlabel('Code');
ylabel('INL (LSB)');
title(sprintf('Integral Nonlinearity | Peak INL = %.2f LSB', max(abs(INL))));
grid on;
xlim([min(code), max(code)]);
set(gca, 'FontSize', 12);

% Subplot 2: DNL
subplot(3,1,2);
plot(code, DNL, 'r-', 'LineWidth', 1.5);
hold on;
yline(0, 'k--', 'LineWidth', 1);
yline(-1, 'r--', 'LineWidth', 1, 'Label', 'Missing Code Threshold');
xlabel('Code');
ylabel('DNL (LSB)');
title(sprintf('Differential Nonlinearity | Peak DNL = %.2f LSB', max(abs(DNL))));
grid on;
xlim([min(code), max(code)]);
set(gca, 'FontSize', 12);

% Subplot 3: Histogram
subplot(3,1,3);
histogram(postCal, 50, 'FaceColor', [0.3, 0.6, 0.9]);
xlabel('Code Value');
ylabel('Count');
title('Code Histogram (Sinewave Input)');
grid on;
set(gca, 'FontSize', 12);

sgtitle('INL/DNL Analysis from Sinewave Histogram', 'FontSize', 14, 'FontWeight', 'bold');

exportgraphics(gcf, fullfile(outDir, 'inl_dnl_analysis.png'), 'Resolution', 300);
close(gcf);

%% Figure 2: Good vs Bad Comparison
fprintf('  [2/2] good_vs_bad_comparison.png...\n');

% Good ADC
bits_ideal = readmatrix(fullfile(dataDir, 'ideal_10bit_sine.csv'));
[~, ~, postCal_good] = FGCalSine(bits_ideal);
[INL_good, DNL_good, code_good] = INLsine(postCal_good, 0.01);

% Bad ADC (with nonlinearity)
[INL_bad, DNL_bad, code_bad] = INLsine(postCal, 0.01);

figure('Position', [100, 100, 1200, 800], 'Visible', 'off');

% INL comparison
subplot(2,2,1);
plot(code_good, INL_good, 'g-', 'LineWidth', 1.5); hold on;
plot(code_bad, INL_bad, 'r-', 'LineWidth', 1.5);
yline(0, 'k--');
xlabel('Code'); ylabel('INL (LSB)');
title('INL Comparison');
legend('Good ADC', 'Bad ADC', 'Location', 'best');
grid on;
set(gca, 'FontSize', 11);

% DNL comparison
subplot(2,2,2);
plot(code_good, DNL_good, 'g-', 'LineWidth', 1.5); hold on;
plot(code_bad, DNL_bad, 'r-', 'LineWidth', 1.5);
yline(0, 'k--'); yline(-1, 'k--', 'LineWidth', 0.5);
xlabel('Code'); ylabel('DNL (LSB)');
title('DNL Comparison');
legend('Good ADC', 'Bad ADC', 'Location', 'best');
grid on;
set(gca, 'FontSize', 11);

% INL histogram
subplot(2,2,3);
histogram(INL_good, 20, 'FaceAlpha', 0.5, 'FaceColor', 'g'); hold on;
histogram(INL_bad, 20, 'FaceAlpha', 0.5, 'FaceColor', 'r');
xlabel('INL (LSB)'); ylabel('Frequency');
title('INL Distribution');
legend('Good ADC', 'Bad ADC');
grid on;
set(gca, 'FontSize', 11);

% DNL histogram
subplot(2,2,4);
histogram(DNL_good, 20, 'FaceAlpha', 0.5, 'FaceColor', 'g'); hold on;
histogram(DNL_bad, 20, 'FaceAlpha', 0.5, 'FaceColor', 'r');
xlabel('DNL (LSB)'); ylabel('Frequency');
title('DNL Distribution');
legend('Good ADC', 'Bad ADC');
grid on;
set(gca, 'FontSize', 11);

sgtitle(sprintf('Good ADC (max|INL|=%.2f) vs Bad ADC (max|INL|=%.2f)', ...
    max(abs(INL_good)), max(abs(INL_bad))), ...
    'FontSize', 14, 'FontWeight', 'bold');

exportgraphics(gcf, fullfile(outDir, 'good_vs_bad_comparison.png'), 'Resolution', 300);
close(gcf);

fprintf('  INLsine figures complete.\n\n');

end
