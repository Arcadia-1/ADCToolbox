function toolset_dout(bits, outputDir, varargin)
%TOOLSET_DOUT Run 6 analysis tools on ADC digital output.
%
%   This script operates in "Fail-Fast" mode. If any tool or calculation
%   fails (e.g., convergence issues, dimension mismatch), the script
%   terminates immediately with a standard MATLAB error.
%
% Inputs:
%   bits       - Digital bits (N samples x B bits, MSB to LSB)
%   outputDir  - Directory to save output figures
%   varargin   - Optional: 'Visible' (T/F), 'Order' (int), 'Prefix' (char)

% --- 1. Input Parsing & Setup ---
p = inputParser;
addParameter(p, 'Visible', false);
addParameter(p, 'Order', 5);
addParameter(p, 'Prefix', 'dout');
parse(p, varargin{:});
opts = p.Results;

if ~exist(outputDir, 'dir'), mkdir(outputDir); end
figVis = opts.Visible;
nBits = size(bits, 2);

% --- 2. Critical Pre-calculation ---
% We calculate calibration weights once.
% If FGCalSine fails here, the script stops immediately.
[w_cal, ~, ~, ~, ~, f_cal] = FGCalSine(bits, 'freq', 0, 'order', opts.Order);

% --- 3. Run Tools Sequentially ---

% Tool 1: Nominal Spectrum (Ideal weights 2^N...2^0)
fprintf('[1/6] Spectrum (Nominal)...');
f = figure('Visible', figVis, 'Position', [100, 100, 800, 600]);
digitalCodes = bits * (2.^(nBits - 1:-1:0))';
specPlot(digitalCodes, 'label', 1, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);
save_and_close(f, outputDir, opts.Prefix, '1_spectrum_nominal', 'Spectrum (Nominal)');

% Tool 2: Calibrated Spectrum (Uses calculated weights)
fprintf('[2/6] Spectrum (Calibrated)...');
f = figure('Visible', figVis, 'Position', [100, 100, 800, 600]);
digitalCodes_cal = bits * w_cal';
specPlot(digitalCodes_cal, 'label', 1, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);
save_and_close(f, outputDir, opts.Prefix, '2_spectrum_calibrated', 'Spectrum (Calibrated)');

% Tool 3: Bit Activity (Toggle rate analysis)
fprintf('[3/6] Bit Activity...');
f = figure('Visible', figVis, 'Position', [100, 100, 800, 600]);
bitActivity(bits, 'AnnotateExtremes', true);
save_and_close(f, outputDir, opts.Prefix, '3_bitActivity', 'Bit Activity');

% Tool 4: Overflow Check (Decomposition check)
fprintf('[4/6] Overflow Check...');
f = figure('Visible', figVis, 'Position', [100, 100, 800, 600]);
overflowChk(bits, w_cal);
save_and_close(f, outputDir, opts.Prefix, '4_overflowChk', 'Overflow Check');

% Tool 5: Weight Scaling (Radix analysis)
fprintf('[5/6] Weight Scaling...');
f = figure('Visible', figVis, 'Position', [100, 100, 800, 600]);
weightScaling(w_cal);
save_and_close(f, outputDir, opts.Prefix, '5_weightScaling', 'Weight Scaling');

% Tool 6: ENoB Sweep (Performance vs Number of Bits)
fprintf('[6/6] ENoB Sweep...');
f = figure('Visible', figVis, 'Position', [100, 100, 800, 600]);
ENoB_bitSweep(bits, 'freq', f_cal, 'order', opts.Order, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);
save_and_close(f, outputDir, opts.Prefix, '6_ENoB_sweep', 'ENoB Bit Sweep');

% --- 4. Generate Summary Panel ---
fprintf('[Panel] Assembling summary... ');

fileNames = {; ...
    '1_spectrum_nominal'; '2_spectrum_calibrated'; '3_bitActivity'; ...
    '4_overflowChk'; '5_weightScaling'; '6_ENoB_sweep'; ...
    };

fig = figure('Visible', figVis, 'Position', [50, 50, 1200, 1600]);
t = tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:6
    nexttile;
    imgName = fullfile(outputDir, sprintf('%s_%s.png', opts.Prefix, fileNames{i}));
    % Since we are in fail-fast mode, we assume files exist.
    imshow(imread(imgName), 'Border', 'tight');
    title(fileNames{i}, 'Interpreter', 'none');
end

sgtitle(['DOUT Analysis: ', opts.Prefix], 'Interpreter', 'none', 'FontSize', 16);
panelPath = fullfile(outputDir, sprintf('PANEL_%s.png', upper(opts.Prefix)));
exportgraphics(fig, panelPath, 'Resolution', 300);
close(fig);

fprintf('Done.\nSaved to: %s\n', panelPath);

end
%% === Helper Function ===
function save_and_close(fig, outDir, prefix, suffix, titleStr)
% Standardizes title formatting, saving, and closing of figures
title(titleStr);
set(gca, 'FontSize', 14);

fname = fullfile(outDir, sprintf('%s_%s.png', prefix, suffix));
exportgraphics(fig, fname, 'Resolution', 150);
close(fig);

fprintf(' OK\n');
end
