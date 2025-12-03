function plot_files = toolset_aout(aout_data, outputDir, varargin)
%TOOLSET_AOUT Run 9 analog analysis tools on calibrated ADC data
%
%   plot_files = toolset_aout(aout_data, outputDir)
%   plot_files = toolset_aout(aout_data, outputDir, 'Visible', true)
%
% Inputs:
%   aout_data  - Analog output signal (1D vector)
%   outputDir  - Directory to save output figures
%
% Optional Parameters:
%   'Visible'   - Show figures (default: false)
%   'Resolution' - ADC resolution in bits (default: 11)
%   'Prefix'    - Filename prefix (default: 'aout')
%
% Outputs:
%   plot_files - Cell array of generated PNG file paths (9x1)
%
% Example:
%   aout_data = readmatrix('sinewave_jitter.csv');
%   plot_files = toolset_aout(aout_data, 'output/test1');

% Parse inputs
p = inputParser;
addParameter(p, 'Visible', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'Resolution', 11, @isnumeric);
addParameter(p, 'Prefix', 'aout', @ischar);
parse(p, varargin{:});
opts = p.Results;

% Pre-compute common parameters
freqCal = findfreq(aout_data);
FullScale = max(aout_data) - min(aout_data);
err_data = aout_data - sinfit(aout_data);

if ~isfolder(outputDir), mkdir(outputDir); end

plot_files = cell(9, 1);


%% Tool 1: tomdec
fprintf('[1/9] tomdec');
figure('Position', [100, 100, 800, 600], 'Visible', opts.Visible);
tomdec(aout_data, freqCal, 10, 1);
sgtitle('tomdec: Time-domain Error Decomposition', 'FontWeight', 'bold');
set(gca, 'FontSize', 14);
plot_files{1} = fullfile(outputDir, sprintf('%s_1_tomdec.png', opts.Prefix));
saveas(gcf, plot_files{1});
close(gcf);
fprintf(' -> %s\n', plot_files{1});

%% Tool 2: plotspec
fprintf('[2/9] plotspec');
figure('Position', [100, 100, 800, 600], 'Visible', opts.Visible);
plotspec(aout_data, 'label', 1, 'harmonic', 5, 'OSR', 1, 'window', @hann);
title('plotspec: Frequency Spectrum');
set(gca, 'FontSize', 14);
plot_files{2} = fullfile(outputDir, sprintf('%s_2_plotspec.png', opts.Prefix));
saveas(gcf, plot_files{2});
close(gcf);
fprintf(' -> %s\n', plot_files{2});

%% Tool 3: plotphase
fprintf('[3/9] plotphase');
figure('Position', [100, 100, 800, 600], 'Visible', opts.Visible);
plotphase(aout_data, 'harmonic', 10, 'mode', 'FFT');
title('plotphase: Phase-domain Error');
set(gca, 'FontSize', 14);
plot_files{3} = fullfile(outputDir, sprintf('%s_3_plotphase.png', opts.Prefix));
saveas(gcf, plot_files{3});
close(gcf);
fprintf(' -> %s\n', plot_files{3});

%% Tool 4: errsin (code)
fprintf('[4/9] errsin_code');
figure('Position', [100, 100, 800, 600], 'Visible', opts.Visible);
errsin(aout_data, 'bin', 20, 'fin', freqCal, 'disp', 1, 'xaxis', 'value');
sgtitle('errsin (code): Error Histogram by Code', 'FontWeight', 'bold');
set(gca, 'FontSize', 14);
plot_files{4} = fullfile(outputDir, sprintf('%s_4_errsin_code.png', opts.Prefix));
saveas(gcf, plot_files{4});
close(gcf);
fprintf(' -> %s\n', plot_files{4});

%% Tool 5: errsin (phase)
fprintf('[5/9] errsin_phase');
figure('Position', [100, 100, 800, 600], 'Visible', opts.Visible);
errsin(aout_data, 'bin', 99, 'fin', freqCal, 'disp', 1, 'xaxis', 'phase');
sgtitle('errsin (phase): Error Histogram by Phase', 'FontWeight', 'bold');
set(gca, 'FontSize', 14);
plot_files{5} = fullfile(outputDir, sprintf('%s_5_errsin_phase.png', opts.Prefix));
saveas(gcf, plot_files{5});
close(gcf);
fprintf(' -> %s\n', plot_files{5});

%% Tool 6: errPDF
fprintf('[6/9] errPDF');
figure('Position', [100, 100, 800, 600], 'Visible', opts.Visible);
errpdf(err_data, 'Resolution', opts.Resolution, 'FullScale', FullScale);
title('errPDF: Error PDF');
set(gca, 'FontSize', 14);
plot_files{6} = fullfile(outputDir, sprintf('%s_6_errPDF.png', opts.Prefix));
saveas(gcf, plot_files{6});
close(gcf);
fprintf(' -> %s\n', plot_files{6});

%% Tool 7: errAutoCorrelation
fprintf('[7/9] errAutoCorrelation');
figure('Position', [100, 100, 800, 600], 'Visible', opts.Visible);
errac(err_data, 'MaxLag', 200, 'Normalize', true);
title('errAutoCorrelation: Error Autocorrelation');
set(gca, 'FontSize', 14);
plot_files{7} = fullfile(outputDir, sprintf('%s_7_errAutoCorrelation.png', opts.Prefix));
saveas(gcf, plot_files{7});
close(gcf);
fprintf(' -> %s\n', plot_files{7});

%% Tool 8: errSpectrum
fprintf('[8/9] errSpectrum');
figure('Position', [100, 100, 800, 600], 'Visible', opts.Visible);
plotspec(err_data, 'label', 0);
title('errSpectrum: Error Spectrum');
set(gca, 'FontSize', 14);
plot_files{8} = fullfile(outputDir, sprintf('%s_8_errSpectrum.png', opts.Prefix));
saveas(gcf, plot_files{8});
close(gcf);
fprintf(' -> %s\n', plot_files{8});

%% Tool 9: errEnvelopeSpectrum
fprintf('[9/9] errEnvelopeSpectrum');
figure('Position', [100, 100, 800, 600], 'Visible', opts.Visible);
errevspec(err_data, 'Fs', 1);
title('errEnvelopeSpectrum: Error Envelope Spectrum');
set(gca, 'FontSize', 14);
plot_files{9} = fullfile(outputDir, sprintf('%s_9_errEnvelopeSpectrum.png', opts.Prefix));
saveas(gcf, plot_files{9});
close(gcf);
fprintf(' -> %s\n', plot_files{9});

fprintf('=== Toolset complete: 9/9 tools completed ===\n\n');
end
