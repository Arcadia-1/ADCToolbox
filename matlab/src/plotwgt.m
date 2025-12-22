function radix = plotwgt(weights)
%PLOTWGT Visualize absolute bit weights with radix annotations
%   This function creates a visualization of ADC bit weights and calculates
%   the radix (scaling factor) between consecutive bits. It helps identify
%   the ADC architecture and detect calibration errors.
%
%   Syntax:
%     radix = PLOTWGT(weights)
%
%   Inputs:
%     weights - Bit weights from MSB to LSB
%       Vector (1 x B), where B is the number of bits
%       Typically obtained from calibration functions like wcalsine
%
%   Outputs:
%     radix - Radix between consecutive bits
%       Vector (1 x B-1), where B is the number of bits
%       radix(i) = |weight(i) / weight(i+1)|
%       Pure binary ADC: radix = 2.00 for all bits
%       Sub-radix ADC: radix < 2.00 (e.g., 1.5-bit/stage -> ~1.90)
%
%   Examples:
%     % Visualize ideal binary weights
%     weights_ideal = 2.^(11:-1:0);  % 12-bit binary
%     radix = plotwgt(weights_ideal);
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
%
%   See also: cdacwgt, wcalsine

% Identify negative weights before taking absolute value
nBits = length(weights);
isNegative = weights < 0;
absWeights = abs(weights);

% Calculate radix between consecutive bits: radix(i) = |weight(i)/weight(i+1)|
radix = zeros(1, nBits-1);
for i = 1:nBits-1
    radix(i) = absWeights(i) / absWeights(i+1);
end

% Create plot with markers showing absolute weights
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

xlabel(sprintf('Bit Index (MSB=%d, LSB=1)', nBits), 'FontSize', 14);
ylabel('Absolute Weight', 'FontSize', 14);
title('Bit Weights with Radix', 'FontSize', 16);
grid on;
xlim([0.5 nBits+0.5]);
set(gca, 'FontSize', 14);
set(gca, 'YScale', 'log');  % Log scale for better visualization

% Reverse x-axis tick labels (MSB=nBits on left, LSB=1 on right)
xticks(1:nBits);
xticklabels(arrayfun(@num2str, nBits:-1:1, 'UniformOutput', false));

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
