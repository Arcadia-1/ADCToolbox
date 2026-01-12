function plot_files = toolset_dout(bits, outputDir, varargin)
%TOOLSET_DOUT Run 6 analysis tools on ADC digital output
%
%   plot_files = toolset_dout(bits, outputDir)
%   plot_files = toolset_dout(bits, outputDir, 'Visible', true)
%
% Inputs:
%   bits       - Digital bits (N samples x B bits, MSB to LSB)
%   outputDir  - Directory to save output figures
%
% Optional Parameters:
%   'Visible' - Show figures (default: false)
%   'Order'   - Polynomial order for FGCalSine (default: 5)
%   'Prefix'  - Filename prefix (default: 'dout')
%
% Outputs:
%   plot_files - Cell array of generated PNG file paths (6x1)
%
% Example:
%   bits = readmatrix('SAR_12b_bits.csv');
%   plot_files = toolset_dout(bits, 'output/test1');

p = inputParser;
addParameter(p, 'Visible', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'Order', 5, @isnumeric);
addParameter(p, 'Prefix', 'dout', @ischar);
parse(p, varargin{:});

if ~isfolder(outputDir), mkdir(outputDir); end
nBits = size(bits, 2);
plot_files = cell(6, 1);

[w_cal, ~, ~, ~, ~, f_cal] = wcalsin(bits, 'freq', 0, 'order', p.Results.Order, 'verbose', 0);

digitalCodes = bits * (2.^(nBits - 1:-1:0))';
digitalCodes_cal = bits * w_cal';

tools = {
    struct('idx', 1, 'name', 'Spectrum (Nominal)', 'suffix', '1_spectrum_nominal', ...
        'pos', [100, 100, 800, 600], 'fn', @() plotspec(digitalCodes, 'label', 1, 'harmonic', 5, 'OSR', 1, 'window', @hann));
    struct('idx', 2, 'name', 'Spectrum (Calibrated)', 'suffix', '2_spectrum_calibrated', ...
        'pos', [100, 100, 800, 600], 'fn', @() plotspec(digitalCodes_cal, 'label', 1, 'harmonic', 5, 'OSR', 1, 'window', @hann));
    struct('idx', 3, 'name', 'Bit Activity', 'suffix', '3_bitActivity', ...
        'pos', [100, 100, 1000, 750], 'fn', @() bitact(bits, 'annotateExtremes', true));
    struct('idx', 4, 'name', 'Overflow Check', 'suffix', '4_overflowChk', ...
        'pos', [100, 100, 1000, 600], 'fn', @() ovfchk(bits, w_cal));
    struct('idx', 5, 'name', 'Weight Scaling', 'suffix', '5_weightScaling', ...
        'pos', [100, 100, 800, 600], 'fn', @() weightScaling(w_cal));
    struct('idx', 6, 'name', 'ENoB Sweep', 'suffix', '6_ENoB_sweep', ...
        'pos', [100, 100, 800, 600], 'fn', @() bitsweep(bits, 'freq', f_cal, 'order', p.Results.Order, 'harmonic', 5, 'OSR', 1, 'winType', @hamming));
};

for i = 1:6
    fprintf('[%d/6] %s', i, tools{i}.name);
    figure('Position', tools{i}.pos, 'Visible', p.Results.Visible);
    tools{i}.fn();
    title(tools{i}.name);
    set(gca, 'FontSize', 14);
    plot_files{i} = fullfile(outputDir, sprintf('%s_%s.png', p.Results.Prefix, tools{i}.suffix));
    exportgraphics(gcf, plot_files{i}, 'Resolution', 150);
    close(gcf);
    fprintf(' -> %s\n', plot_files{i});
end

fprintf('=== Toolset complete: 6/6 tools completed ===\n\n');
end
