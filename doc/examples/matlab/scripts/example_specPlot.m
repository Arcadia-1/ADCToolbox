function example_specPlot()
%% example_specPlot - Generate documentation figures for specPlot
%
% Generates figures showing:
% 1. Ideal spectrum (clean sinewave)
% 2. Spectrum with harmonic distortion
% 3. Comparison: before vs after calibration

outDir = '../../figures/specPlot';
if ~isfolder(outDir), mkdir(outDir); end

dataDir = '../../data';

fprintf('Generating specPlot figures...\n');

%% Figure 1: Ideal Spectrum
fprintf('  [1/3] ideal_spectrum.png...\n');

bits_ideal = readmatrix(fullfile(dataDir, 'ideal_10bit_sine.csv'));
[~, ~, postCal] = FGCalSine(bits_ideal);

figure('Position', [100, 100, 1000, 700], 'Visible', 'off');
[ENoB, SNDR, SFDR, SNR, THD] = specPlot(postCal, 'label', 1, 'harmonic', 5);
title('Ideal ADC Spectrum (No Distortion)');
set(gca, 'FontSize', 12);

% Add text box with metrics
annotation('textbox', [0.15, 0.75, 0.2, 0.15], ...
    'String', {sprintf('ENoB: %.2f bits', ENoB), ...
               sprintf('SNDR: %.2f dB', SNDR), ...
               sprintf('SFDR: %.2f dB', SFDR), ...
               sprintf('THD: %.2f dB', THD), ...
               sprintf('SNR: %.2f dB', SNR)}, ...
    'FontSize', 11, 'BackgroundColor', 'white', 'EdgeColor', 'black');

exportgraphics(gcf, fullfile(outDir, 'ideal_spectrum.png'), 'Resolution', 300);
close(gcf);

%% Figure 2: Spectrum with Harmonic Distortion
fprintf('  [2/3] with_distortion.png...\n');

bits_distortion = readmatrix(fullfile(dataDir, 'with_harmonic_distortion.csv'));
[~, ~, postCal_dist] = FGCalSine(bits_distortion);

figure('Position', [100, 100, 1000, 700], 'Visible', 'off');
[ENoB_d, SNDR_d, SFDR_d, SNR_d, THD_d] = specPlot(postCal_dist, 'label', 1, 'harmonic', 5);
title('Spectrum with Harmonic Distortion (HD2, HD3)');
set(gca, 'FontSize', 12);

% Add text box
annotation('textbox', [0.15, 0.75, 0.2, 0.15], ...
    'String', {sprintf('ENoB: %.2f bits', ENoB_d), ...
               sprintf('SNDR: %.2f dB', SNDR_d), ...
               sprintf('SFDR: %.2f dB', SFDR_d), ...
               sprintf('THD: %.2f dB (HD2+HD3)', THD_d), ...
               sprintf('SNR: %.2f dB', SNR_d)}, ...
    'FontSize', 11, 'BackgroundColor', 'white', 'EdgeColor', 'black');

exportgraphics(gcf, fullfile(outDir, 'with_distortion.png'), 'Resolution', 300);
close(gcf);

%% Figure 3: Before vs After Calibration
fprintf('  [3/3] before_after_calibration.png...\n');

bits = readmatrix(fullfile(dataDir, 'with_inl_dnl.csv'));

% Before calibration: use nominal weights
nominalWeights = 2.^(9:-1:0);
signal_before = bits * nominalWeights';

% After calibration
[~, ~, signal_after] = FGCalSine(bits);

figure('Position', [100, 100, 1200, 500], 'Visible', 'off');

% Before
subplot(1,2,1);
[ENoB_before, SNDR_before, SFDR_before] = specPlot(signal_before, 'label', 1, 'harmonic', 5);
title('Before Calibration (Nominal Weights)');
set(gca, 'FontSize', 11);

% After
subplot(1,2,2);
[ENoB_after, SNDR_after, SFDR_after] = specPlot(signal_after, 'label', 1, 'harmonic', 5);
title('After FGCalSine Calibration');
set(gca, 'FontSize', 11);

% Overall title
sgtitle(sprintf('Calibration Improvement: ΔENoB = %.2f bits, ΔSNDR = %.2f dB', ...
    ENoB_after - ENoB_before, SNDR_after - SNDR_before), ...
    'FontSize', 14, 'FontWeight', 'bold');

exportgraphics(gcf, fullfile(outDir, 'before_after_calibration.png'), 'Resolution', 300);
close(gcf);

fprintf('  specPlot figures complete.\n\n');

end
