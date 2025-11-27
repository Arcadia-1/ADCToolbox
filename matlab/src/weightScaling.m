function radix = weightScaling(weights)
%WEIGHTSCALING Visualize absolute bit weights with radix annotations
%
%   radix = weightScaling(weights)
%
% Inputs:
%   weights    - Bit weights (1 x B array), from MSB to LSB
%
% Outputs:
%   radix      - Radix between consecutive bits (weight[i-1]/weight[i])
%
% Description:
%   Visualizes absolute bit weight values with annotations showing the
%   radix (scaling factor) relative to the previous bit.
%
%   Pure binary weights have radix = 2.00 (each bit is half the previous).
%   Architectures with redundancy or sub-radix may deviate from 2.00.
%
%   What to look for:
%   - Radix = 2.00: Binary scaling (SAR, pure binary)
%   - Radix < 2.00: Redundancy or sub-radix (e.g., 1.5-bit/stage → ~1.90)
%   - Radix > 2.00: Unusual, may indicate calibration error
%   - Consistent pattern: Expected architecture behavior
%   - Random jumps: Calibration errors or bit mismatch
%
% Example:
%   bits = readmatrix('dout_SAR_12b.csv');
%   [weight_cal, ~, ~, ~, ~, ~] = FGCalSine(bits, 'freq', 0, 'order', 5);
%   radix = weightScaling(weight_cal);

% Calculate radix between consecutive bits (weight[i-1] / weight[i])
nBits = length(weights);
radix = zeros(1, nBits);
radix(1) = NaN;  % No radix for first bit
for i = 2:nBits
    radix(i) = weights(i-1) / weights(i);
end

% Create line plot with markers showing absolute weights
hold on;
plot(1:nBits, weights, '-o', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.3 0.6 0.8], 'Color', [0.3 0.6 0.8]);
xlabel('Bit Index (1=MSB, N=LSB)', 'FontSize', 14);
ylabel('Absolute Weight', 'FontSize', 14);
title('Bit Weights with Radix', 'FontSize', 16);
grid on;
xlim([0.5 nBits+0.5]);
set(gca, 'FontSize', 14);
set(gca, 'YScale', 'log');  % Log scale for better visualization

% Annotate radix on top of each data point (except first bit)
for b = 2:nBits
    y_pos = weights(b) * 1.5;  % Position text above the marker
    text(b, y_pos, sprintf('/%.2f', radix(b)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, ...
        'Color', [0.2 0.2 0.2], 'FontWeight', 'bold');
end
hold off;

end
