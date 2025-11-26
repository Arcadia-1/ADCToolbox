function bit_usage = bitActivity(bits, varargin)
%BITACTIVITY Analyze and plot the percentage of 1's in each bit
%
%   bit_usage = bitActivity(bits)
%   bit_usage = bitActivity(bits, 'AnnotateExtremes', true)
%
% Inputs:
%   bits       - Binary matrix (N x B), N=samples, B=bits (MSB to LSB)
%
% Optional Parameters:
%   'AnnotateExtremes' - Annotate bits with >95% or <5% activity (default: true)
%
% Outputs:
%   bit_usage  - Percentage of 1's for each bit (1 x B array)
%
% Description:
%   This function calculates and visualizes the percentage of 1's in each bit
%   position. A bar chart is created with a reference line at 50% (ideal).
%
%   What to look for:
%   - ~50%: Good bit activity, well-utilized
%   - >95%: Bit stuck high or large positive DC offset
%   - <5%:  Bit stuck low or large negative DC offset
%   - Gradual trend: Indicates DC offset pattern across MSB→LSB
%
% Example:
%   bits = readmatrix('dout_SAR_12b.csv');
%   bit_usage = bitActivity(bits);

% Parse optional inputs
p = inputParser;
addParameter(p, 'AnnotateExtremes', true, @islogical);
parse(p, varargin{:});
annotate = p.Results.AnnotateExtremes;

% Calculate percentage of 1's for each bit
nBits = size(bits, 2);
bit_usage = mean(bits, 1) * 100;  % Percentage of 1's per bit

% Create bar chart
bar(1:nBits, bit_usage, 'FaceColor', [0.2 0.4 0.8]);
hold on;
yline(50, 'r--', 'LineWidth', 2, 'Label', 'Ideal (50%)');
xlabel('Bit Index (1=MSB, N=LSB)', 'FontSize', 14);
ylabel('Percentage of 1''s (%)', 'FontSize', 14);
title('Bit Activity Analysis', 'FontSize', 16);
ylim([0 100]);
xlim([0.5 nBits+0.5]);
grid on;
set(gca, 'FontSize', 14);

% Add text annotations for extreme values
if annotate
    for b = 1:nBits
        if bit_usage(b) > 95
            text(b, bit_usage(b) + 3, sprintf('%.1f%%', bit_usage(b)), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, ...
                'Color', 'red', 'FontWeight', 'bold');
        elseif bit_usage(b) < 5
            text(b, bit_usage(b) + 3, sprintf('%.1f%%', bit_usage(b)), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, ...
                'Color', 'red', 'FontWeight', 'bold');
        end
    end
end

hold off;

end
