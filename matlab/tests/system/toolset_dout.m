function results = toolset_dout(bits, weight, outputDir, varargin)
%TOOLSET_DOUT Run 4 digital analysis tools on ADC digital output
%
%   results = toolset_dout(bits, weight, outputDir, options)
%
% Inputs:
%   bits       - Binary matrix (N x B), N=samples, B=bits (MSB to LSB)
%   weight     - Calibrated bit weights (1 x B array)
%   outputDir  - Directory to save output figures
%   options    - Optional name-value pairs:
%                'SaveFigs' (default: true) - Save individual figures
%                'CreatePanel' (default: true) - Create overview panel
%                'Visible' (default: false) - Figure visibility
%                'Resolution' (default: 10) - ADC resolution (bits)
%                'FreqCal' (default: auto) - Calibrated frequency for ENoB sweep
%                'Order' (default: 5) - Polynomial order for calibration
%
% Outputs:
%   results    - Struct containing all analysis results

% Parse optional inputs
p = inputParser;
addParameter(p, 'SaveFigs', true, @islogical);
addParameter(p, 'CreatePanel', true, @islogical);
addParameter(p, 'Visible', false, @islogical);
addParameter(p, 'Resolution', 10, @isnumeric);
addParameter(p, 'FreqCal', [], @isnumeric);
addParameter(p, 'Order', 5, @isnumeric);
parse(p, varargin{:});
opts = p.Results;

% Set figure visibility
if opts.Visible
    figVis = 'on';
else
    figVis = 'off';
end

% Initialize results structure
results = struct();

% Create output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fprintf('\n=== Running DOUT Toolset (4 Digital Analysis Tools) ===\n');

% Calculate digital codes using nominal and calibrated weights
nominalWeights = 2.^(opts.Resolution-1:-1:0);
digitalCodes_nominal = bits * nominalWeights';
digitalCodes_calibrated = bits * weight';

% Store digital codes in results
results.digitalCodes_nominal = digitalCodes_nominal;
results.digitalCodes_calibrated = digitalCodes_calibrated;

% -------------------------------------------------------------------------
% Tool 1: Digital Spectrum with Nominal Weights
% -------------------------------------------------------------------------
fprintf('  (1/4) Running specPlot on digital signal (nominal weights)...\n');
try
    figure('Position', [100, 100, 1200, 800], 'Visible', figVis);
    [ENoB_nom, SNDR_nom, SFDR_nom, SNR_nom, THD_nom, ~, ~, ~] = ...
        specPlot(digitalCodes_nominal, 'label', 1, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);
    title('Digital Signal Spectrum (Nominal Binary Weights)');

    if opts.SaveFigs
        saveas(gcf, fullfile(outputDir, 'dout_1_spectrum_nominal.png'));
        fprintf('      [Saved] dout_1_spectrum_nominal.png\n');
    end
    close(gcf);

    results.nominal.ENoB = ENoB_nom;
    results.nominal.SNDR = SNDR_nom;
    results.nominal.SFDR = SFDR_nom;
    results.nominal.SNR = SNR_nom;
    results.nominal.THD = THD_nom;
    fprintf('      ENoB=%.2f, SNDR=%.2f dB, SFDR=%.2f dB\n', ENoB_nom, SNDR_nom, SFDR_nom);
catch ME
    fprintf('      [Error] %s\n', ME.message);
    results.nominal.error_msg = ME.message;
end

% -------------------------------------------------------------------------
% Tool 2: Digital Spectrum with Calibrated Weights
% -------------------------------------------------------------------------
fprintf('  (2/4) Running specPlot on calibrated digital signal...\n');
try
    figure('Position', [100, 100, 1200, 800], 'Visible', figVis);
    [ENoB_cal, SNDR_cal, SFDR_cal, SNR_cal, THD_cal, ~, ~, ~] = ...
        specPlot(digitalCodes_calibrated, 'label', 1, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);
    title('Digital Signal Spectrum (Calibrated Weights)');

    if opts.SaveFigs
        saveas(gcf, fullfile(outputDir, 'dout_2_spectrum_calibrated.png'));
        fprintf('      [Saved] dout_2_spectrum_calibrated.png\n');
    end
    close(gcf);

    results.calibrated.ENoB = ENoB_cal;
    results.calibrated.SNDR = SNDR_cal;
    results.calibrated.SFDR = SFDR_cal;
    results.calibrated.SNR = SNR_cal;
    results.calibrated.THD = THD_cal;
    fprintf('      ENoB=%.2f, SNDR=%.2f dB, SFDR=%.2f dB\n', ENoB_cal, SNDR_cal, SFDR_cal);

    % Calculate improvement
    if isfield(results.nominal, 'ENoB')
        improvement_ENoB = ENoB_cal - results.nominal.ENoB;
        improvement_SNDR = SNDR_cal - results.nominal.SNDR;
        fprintf('      Improvement: +%.2f ENoB, +%.2f dB SNDR\n', improvement_ENoB, improvement_SNDR);
        results.improvement.ENoB = improvement_ENoB;
        results.improvement.SNDR = improvement_SNDR;
    end
catch ME
    fprintf('      [Error] %s\n', ME.message);
    results.calibrated.error_msg = ME.message;
end

% -------------------------------------------------------------------------
% Tool 3: ENoB Bit Sweep
% -------------------------------------------------------------------------
fprintf('  (3/4) Running ENoB_bitSweep...\n');
try
    % Determine frequency for calibration
    if isempty(opts.FreqCal)
        % Auto-detect frequency using FGCalSine
        fprintf('      Auto-detecting frequency...\n');
        [~, ~, ~, ~, ~, freqCal] = FGCalSine(bits, 'freq', 0, 'order', opts.Order);
        fprintf('      Detected frequency: %.6f (normalized)\n', freqCal);
    else
        freqCal = opts.FreqCal;
    end

    figure('Position', [100, 100, 1200, 800], 'Visible', figVis);
    [ENoB_sweep, nBits_vec] = ENoB_bitSweep(bits, 'freq', freqCal, ...
        'order', opts.Order, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);
    title('ENoB vs Number of Bits Used for Calibration');

    if opts.SaveFigs
        saveas(gcf, fullfile(outputDir, 'dout_3_ENoB_sweep.png'));
        fprintf('      [Saved] dout_3_ENoB_sweep.png\n');
    end
    close(gcf);

    results.ENoB_sweep.ENoB = ENoB_sweep;
    results.ENoB_sweep.nBits_vec = nBits_vec;
    results.ENoB_sweep.freqCal = freqCal;
    fprintf('      Max ENoB: %.2f at %d bits\n', max(ENoB_sweep), nBits_vec(ENoB_sweep == max(ENoB_sweep)));
catch ME
    fprintf('      [Error] %s\n', ME.message);
    results.ENoB_sweep.error_msg = ME.message;
end

% -------------------------------------------------------------------------
% Tool 4: Overflow Check
% -------------------------------------------------------------------------
fprintf('  (4/4) Running overflowChk...\n');
try
    figure('Position', [100, 100, 1200, 800], 'Visible', figVis);
    data_decom = overflowChk(bits, weight);
    title('Overflow Check: Bit-by-Bit Contribution');

    if opts.SaveFigs
        saveas(gcf, fullfile(outputDir, 'dout_4_overflowChk.png'));
        fprintf('      [Saved] dout_4_overflowChk.png\n');
    end
    close(gcf);

    results.overflowChk.data_decom = data_decom;
    results.overflowChk.completed = true;
catch ME
    fprintf('      [Error] %s\n', ME.message);
    results.overflowChk.error_msg = ME.message;
end

% -------------------------------------------------------------------------
% Create Panel Overview (2x2 grid)
% -------------------------------------------------------------------------
if opts.CreatePanel
    fprintf('\n  Creating DOUT Panel Overview (2x2 grid)...\n');

    plotFiles = {
        fullfile(outputDir, 'dout_1_spectrum_nominal.png');
        fullfile(outputDir, 'dout_2_spectrum_calibrated.png');
        fullfile(outputDir, 'dout_3_ENoB_sweep.png');
        fullfile(outputDir, 'dout_4_overflowChk.png');
    };

    plotLabels = {
        '(1) Digital Spectrum - Nominal Weights';
        '(2) Digital Spectrum - Calibrated Weights';
        '(3) ENoB vs Bits Used for Calibration';
        '(4) Overflow Check';
    };

    fig = figure('Position', [50 50 1600 1200], 'Visible', figVis);
    tlo = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    for p = 1:length(plotFiles)
        nexttile(tlo, p);
        img_path = plotFiles{p};

        try
            if isfile(img_path)
                img = imread(img_path);
                imshow(img, 'Border', 'tight');
                axis tight;
                axis off;
                title(plotLabels{p}, 'FontSize', 12, 'Interpreter', 'none');
            else
                text(0.5, 0.5, sprintf('Missing:\n%s', plotLabels{p}), ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'FontSize', 10, 'Color', 'red');
                axis([0 1 0 1]);
                axis off;
                title(plotLabels{p}, 'FontSize', 12, 'Interpreter', 'none', 'Color', 'red');
            end
        catch ME
            text(0.5, 0.5, sprintf('Error:\n%s', ME.message), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'FontSize', 10, 'Color', 'red');
            axis([0 1 0 1]);
            axis off;
            title(plotLabels{p}, 'FontSize', 12, 'Interpreter', 'none', 'Color', 'red');
        end
    end

    timeStr = datestr(now, 'yyyymmdd_HHMMSS');
    sgtitle(sprintf('DOUT Toolset Overview - %s', timeStr), 'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

    panelPath = fullfile(outputDir, sprintf('PANEL_DOUT_%s.png', timeStr));
    exportgraphics(fig, panelPath, 'Resolution', 300);
    fprintf('    [Saved] %s\n', panelPath);
    close(fig);

    results.panel_path = panelPath;
end

fprintf('\n=== DOUT Toolset Complete ===\n\n');

end
