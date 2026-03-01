function [radix, wgtsca, effres] = plotwgt(weights, disp)
%PLOTWGT Visualize absolute bit weights with radix annotations
%   This function creates a visualization of ADC bit weights and calculates
%   the radix (scaling factor) between consecutive bits. It helps identify
%   the ADC architecture and detect calibration errors.
%
%   Additionally, it computes an optimal weight scaling factor and estimates
%   the effective resolution based on the significant bits.
%
%   Syntax:
%     radix = PLOTWGT(weights)
%     radix = PLOTWGT(weights, disp)
%     [radix, wgtsca] = PLOTWGT(weights)
%     [radix, wgtsca, effres] = PLOTWGT(weights)
%
%   Inputs:
%     weights - Bit weights from MSB to LSB
%       Vector (1 x B), where B is the number of bits
%       Typically obtained from calibration functions like wcalsin
%
%     disp - Display flag (optional, default: 1)
%       1: Enable plotting (default)
%       0: Disable plotting, only compute outputs
%
%   Outputs:
%     radix - Radix between consecutive bits
%       Vector (1 x B-1), where B is the number of bits
%       radix(i) = |weight(i) / weight(i+1)|
%       Pure binary ADC: radix = 2.00 for all bits
%       Sub-radix ADC: radix < 2.00 (e.g., 1.5-bit/stage -> ~1.90)
%
%     wgtsca - Optimal weight scaling factor
%       Scalar that normalizes weights to minimize rounding error
%       Computed by detecting significant bits (ratio jumps < 3) and
%       refining to align weights with integers
%
%     effres - Effective resolution in bits
%       Estimated from significant weights: log2(sum(absW_sig)/absW_LSB + 1)
%       This represents the effective number of bits based on weight ratios
%
%   Algorithm for wgtsca and effres:
%     1. Sort absolute weights descending to identify bit significance
%     2. Find "significant" bits by detecting ratio jumps >= 3
%        (large jumps indicate transition to noise/redundant bits)
%     3. Initial wgtsca normalizes the smallest significant weight to 1
%     4. Refine wgtsca to minimize rounding error across significant weights
%     5. Compute effres as log2(sum(absW_sig)/absW_LSB + 1)
%
%   Examples:
%     % Visualize ideal binary weights
%     weights_ideal = 2.^(11:-1:0);  % 12-bit binary
%     [radix, wgtsca, effres] = plotwgt(weights_ideal);
%
%     % Visualize CDAC weights (6-bit with 3+3 segments)
%     cd = [4 2 1 4 2 1];       % Two 3-bit segments [MSB ... LSB]
%     cb = [0 0 0 8/7 0 0];     % Bridge cap between segments
%     cp = [0 0 0 0 0 1];       % Parasitic at LSB
%     weight = cdacwgt(cd, cb, cp);
%     radix = plotwgt(weight);
%
%   Notes:
%     - Radix = 2.00: Binary scaling (SAR, pure binary)
%     - Radix < 2.00: Redundancy or sub-radix (e.g., 1.5-bit/stage -> ~1.90)
%     - Radix > 2.00: Unusual, may indicate calibration error
%     - Consistent pattern: Expected architecture behavior
%     - Random jumps: Calibration errors or bit mismatch
%     - Y-axis uses logarithmic scale for better visualization
%     - Negative weights are displayed in red color to indicate sign errors
%     - MSB and LSB (smallest significant bit) are marked in the plot
%
%   See also: cdacwgt, wcalsin

% Default value for disp
if nargin < 2
    disp = 1;
end

% Identify negative weights before taking absolute value
nBits = length(weights);
isNegative = weights < 0;
absWeights = abs(weights);

% Calculate radix between consecutive bits: radix(i) = |weight(i)/weight(i+1)|
radix = zeros(1, nBits-1);
for i = 1:nBits-1
    radix(i) = absWeights(i) / absWeights(i+1);
end

%% Compute wgtsca and effres
% Algorithm:
% 1. Sort absolute weights descending to identify bit significance
% 2. Find "significant" bits by detecting ratio jumps >= 3
%    (large jumps indicate transition to noise/redundant bits)
% 3. Initial wgtsca normalizes the smallest significant weight to 1
% 4. Refine wgtsca to minimize rounding error across all weights
% 5. Compute effres as the effective resolution from significant bits

% Step 1: Sort absolute weights from large to small, track original indices
[absW, sortIdx] = sort(absWeights, 'descend');

% Step 2: Calculate ratios between adjacent abs(weights)
ratios = absW(1:end-1) ./ absW(2:end);

% Step 3: Find max K such that ratios(i) < 3 for all i = 1 to K
% K+1 is the number of "significant" bits
idx = find(ratios >= 3, 1, 'first');
if isempty(idx)
    K = length(ratios);  % all ratios < 3, all bits significant
else
    K = idx - 1;
end

% Track original indices of MSB and LSB (smallest significant bit)
msbIdx = sortIdx(1);      % original index of largest weight
lsbIdx = sortIdx(K+1);    % original index of smallest significant weight

% Step 4: Initial scaling - normalize smallest significant weight to 1
wgtsca = 1 / absW(K+1);

% Step 5: Refine wgtsca to minimize rounding error
% Search around the initial estimate (0.5x to 1.5x of MSB weight)
% to find scaling that makes weights closest to integers
absW_sig = absW(1:K+1);  % only use significant weights for refinement
w_err = rms(absW_sig * wgtsca - round(absW_sig * wgtsca));
WMSB_init = round(absW(1) * wgtsca);
WMSB_min = max(1, round(WMSB_init * 0.5));  % ensure >= 1 to avoid zero scaling
WMSB_max = max(WMSB_min, round(WMSB_init * 1.5));
for WMSB = WMSB_min:WMSB_max
    w_refine = WMSB / absW(1);
    w_err_ref = rms(absW_sig * w_refine - round(absW_sig * w_refine));
    if w_err > w_err_ref
        w_err = w_err_ref;
        wgtsca = w_refine;
    end
end

% Step 6: Calculate effective resolution
effres = log2(sum(absW_sig) / absW_sig(end) + 1);

%% Create plot with markers showing absolute weights
if disp
    hold on;

    % Plot connecting line first (black, so markers appear on top)
    plot(1:nBits, absWeights, '-', 'LineWidth', 2, 'Color', [0 0 0], ...
        'HandleVisibility', 'off');

    % Plot positive weight markers (blue)
    posIdx = find(~isNegative);
    hPos = [];
    if ~isempty(posIdx)
        hPos = plot(posIdx, absWeights(posIdx), 'o', 'MarkerSize', 8, ...
            'MarkerFaceColor', [0.3 0.6 0.8], 'Color', [0.3 0.6 0.8], ...
            'LineWidth', 2);
    end

    % Plot negative weight markers (red)
    negIdx = find(isNegative);
    hNeg = [];
    if ~isempty(negIdx)
        hNeg = plot(negIdx, absWeights(negIdx), 'o', 'MarkerSize', 8, ...
            'MarkerFaceColor', [0.9 0.3 0.3], 'Color', [0.9 0.3 0.3], ...
            'LineWidth', 2);
    end

    % Mark MSB (largest weight) with text indicator (red, left of marker)
    text(msbIdx, absWeights(msbIdx)/1.5, 'MSB', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
        'FontSize', 10, 'Color', [0.8 0.1 0.1], 'FontWeight', 'bold');

    % Mark LSB (smallest significant weight) with text indicator (green, left of marker)
    if lsbIdx ~= msbIdx
        text(lsbIdx, absWeights(lsbIdx)/1.5, 'LSB', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
            'FontSize', 10, 'Color', [0.1 0.5 0.1], 'FontWeight', 'bold');
    end

    % Configure primary axes
    ax1 = gca;
    ylabel('Absolute Weight');
    title('Bit Weights with Radix');
    grid on;
    xlim([0.5 nBits+0.5]);
    set(ax1, 'YScale', 'log');  % Log scale for better visualization

    % Extend y-axis range to avoid data points at edges (log scale: use multiplicative margins)
    ymin = min(absWeights(absWeights > 0)) / 4;
    ymax = max(absWeights) * 4;
    ylim([ymin, ymax]);

    % Bottom x-axis: ascending order (1 to nBits) - array order
    xticks(1:nBits);
    xticklabels(arrayfun(@num2str, 1:nBits, 'UniformOutput', false));
    xlabel('Bit Index');

    % Add secondary labels (nBits:-1:1) right above the x-axis tick labels
    for i = 1:nBits
        text(i, ymin * 2, sprintf('%d', nBits - i + 1), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
            'Color', [0.5 0.5 0.5]);
    end

    % Add effective resolution text in upper right corner
    text(0.98, 0.88, sprintf('Eff. Res: %.2f bits', effres), ...
        'Units', 'normalized', 'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'top', 'FontWeight', 'bold', 'Color', [0 0 0], ...
        'BackgroundColor', [1 1 1], 'EdgeColor', [0.5 0.5 0.5]);

    % Add legend for positive and negative weights
    if ~isempty(hPos) && ~isempty(hNeg)
        legend([hPos, hNeg], {'Positive', 'Negative'}, 'Location', 'best');
    elseif ~isempty(hNeg)
        legend(hNeg, {'Negative'}, 'Location', 'best');
    end

    % Annotate radix on top of each data point (except first bit)
    for b = 2:nBits
        y_pos = absWeights(b) * 1.5;  % Position text above the marker
        text(b, y_pos, sprintf('/%.2f', radix(b-1)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, ...
            'Color', [0.2 0.2 0.2], 'FontWeight', 'bold');
    end
    hold off;
end

end
