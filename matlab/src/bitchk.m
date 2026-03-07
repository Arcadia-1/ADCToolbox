function [range_min, range_max, ovf_percent_zero, ovf_percent_one] = bitchk(bits, varargin)
%BITCHK Check ADC overflow by analyzing bit segment residue distributions
%   This function analyzes ADC output data to detect overflow conditions in
%   bit segments. For each bit position, it calculates the normalized residue
%   (sub-code from that bit to LSB) and visualizes the distribution to identify
%   segments that reach their minimum (all 0) or maximum (all 1) limits, indicating
%   potential overflow.
%
%   Syntax:
%     BITCHK(bits)
%     BITCHK(bits, wgt)
%     BITCHK(bits, wgt, chkpos)
%     [range_min, range_max, ovf_percent_zero, ovf_percent_one] = BITCHK(...)
%   or using parameter pairs:
%     BITCHK(bits, 'name', value)
%
%   Inputs:
%     bits - Raw ADC output bit matrix
%       Matrix (N-by-M) where N is number of samples, M is number of bits
%       Each column represents one bit of the ADC output
%       Each row represents the output code for one sample point
%       Code format: [MSB, MSB-1, MSB-2, .., LSB]
%     wgt - Bit weights for ADC code calculation. Optional.
%       Vector (1-by-M)
%       Weight of each bit in the ADC
%       Default: binary weights [2^(M-1), 2^(M-2), ..., 2, 1]
%     chkpos - Bit position to check for overflow. Optional.
%       Scalar
%       Overflow detection based on residue at chkpos-th bit (1 for LSB, M for MSB)
%       Range: [1, M]
%       Default: M (check at MSB)
%     disp - Enable figure display. Optional.
%       Logical
%       Default: true if nargout == 0, false otherwise
%
%   Outputs:
%     range_min - Minimum normalized residue for each bit position
%       Vector (1-by-M)
%       Shows how close each bit segment gets to underflow (0)
%     range_max - Maximum normalized residue for each bit position
%       Vector (1-by-M)
%       Shows how close each bit segment gets to overflow (1)
%     ovf_percent_zero - Percentage of samples at or below 0 for each bit
%       Vector (1-by-M)
%       Underflow percentage per bit position
%     ovf_percent_one - Percentage of samples at or above 1 for each bit
%       Vector (1-by-M)
%       Overflow percentage per bit position
%
%   Plot Description (when disp == true):
%     - X-axis: Bit position (MSB to LSB, labeled M to 1)
%     - Y-axis: Normalized residue distribution [0, 1]
%     - Blue dots: Normal samples (no overflow)
%     - Red dots: Samples with overflow (>= 1)
%     - Green dots: Samples with underflow (<= 0)
%     - Red lines: Min/max range of residue for each bit
%     - Black horizontal lines: Average value of each bit (with value label)
%     - Black lines: Boundaries at 0 and 1
%     - Text: Percentage of samples at 0 (bottom) and 1 (top)
%     - Legend: Explains all displayed elements
%
%   Examples:
%     % Check overflow for 10-bit ADC with default binary weights
%     bits = randi([0 1], 10000, 10);
%     bitchk(bits)
%
%     % Check overflow with custom weights
%     wgt = 2.^(9:-1:0);
%     bitchk(bits, wgt)
%
%     % Check overflow of the segment: from the 8th-bit to LSB
%     bitchk(bits, wgt, 8)
%
%     % Get overflow statistics without display
%     [range_min, range_max, pct_zero, pct_one] = bitchk(bits);
%
%     % Get overflow statistics with forced display
%     [range_min, range_max, pct_zero, pct_one] = bitchk(bits, 'disp', true);
%
%   Notes:
%     - A bit segment is the sub-code formed from one bit to the LSB
%     - Residue is normalized by dividing by the sum of weights in the corresponding segment
%     - Function automatically transposes input if N < M
%     - Uses transparent markers to visualize density of data points
%     - Plot is displayed when disp == true (default: true if no output arguments)
%
%   See also: plot, scatter

[N,M] = size(bits);
if(N < M)
    bits = bits';
    [N,M] = size(bits);
end

% Parse input arguments
p = inputParser;
addOptional(p, 'wgt', 2.^(M-1:-1:0), @(x) isnumeric(x) && isvector(x));
addOptional(p, 'chkpos', M, @(x) isnumeric(x) && isscalar(x) && (x >= 1) && (x <= M));
addParameter(p, 'disp', nargout == 0, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
parse(p, varargin{:});
wgt = p.Results.wgt;
chkpos = p.Results.chkpos;
dispFlag = logical(p.Results.disp);

data_decom = zeros([N,M]);
range_min = zeros([1,M]);
range_max = zeros([1,M]);
ovf_percent_zero = zeros([1,M]);
ovf_percent_one = zeros([1,M]);

for ii = 1:M
    tmp = bits(:,ii:end)*wgt(ii:end)';

    data_decom(:,ii) = tmp / sum(wgt(ii:end));
    range_min(ii) = min(tmp) / sum(wgt(ii:end));
    range_max(ii) = max(tmp) / sum(wgt(ii:end));

    % Calculate overflow percentages
    ovf_percent_zero(ii) = sum(data_decom(:,ii) <= 0) / N * 100;
    ovf_percent_one(ii) = sum(data_decom(:,ii) >= 1) / N * 100;
end

ovf_zero = (data_decom(:,M-chkpos+1) <= 0);
ovf_one = (data_decom(:,M-chkpos+1) >= 1);
non_ovf = ~(ovf_zero | ovf_one);

% Plot if display is enabled
if dispFlag
    hold on;
    hBoundary = plot([0,M+1],[1,1],'-k');
    plot([0,M+1],[0,0],'-k');
    hMinMax = plot((1:M),range_min,'-r');
    plot((1:M),range_max,'-r');

    % Calculate and plot average markers for each bit (bit-wise average)
    avg_bits = mean(bits, 1);
    lineWidth = 0.2;  % Half-width of horizontal line marker
    hAvg = plot([1-lineWidth, 1+lineWidth], [avg_bits(1), avg_bits(1)], '-k', 'LineWidth', 2);
    for ii = 1:M
        plot([ii-lineWidth, ii+lineWidth], [avg_bits(ii), avg_bits(ii)], '-k', 'LineWidth', 2);
        text(ii+lineWidth+0.05, avg_bits(ii), num2str(avg_bits(ii),'%.2f'), 'FontSize', 8);
    end

    % Create dummy scatter points for legend (full opacity, invisible location)
    hNormal = scatter(NaN, NaN, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');
    hOverflow = scatter(NaN, NaN, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r');
    hUnderflow = scatter(NaN, NaN, 'MarkerFaceColor', [0 0.5 0], 'MarkerEdgeColor', [0 0.5 0]);

    % Plot scatter points (transparent for density visualization)
    chkpos_idx = M - chkpos + 1;
    for ii = 1:M

        h = scatter(ones([1,sum(non_ovf)])*ii, data_decom(non_ovf,ii), 'MarkerFaceColor','b','MarkerEdgeColor','b');
        h.MarkerFaceAlpha = min(max(10/N,0.01),1);
        h.MarkerEdgeAlpha = min(max(10/N,0.01),1);

        % Only show overflow/underflow points up to chkpos (not after)
        if ii <= chkpos_idx
            h = scatter(ones([1,sum(ovf_one)])*ii-0.2, data_decom(ovf_one,ii), 'MarkerFaceColor','r','MarkerEdgeColor','r');
            h.MarkerFaceAlpha = min(max(10/sum(ovf_one),0.01),1);
            h.MarkerEdgeAlpha = min(max(10/sum(ovf_one),0.01),1);

            h = scatter(ones([1,sum(ovf_zero)])*ii-0.1, data_decom(ovf_zero,ii), 'MarkerFaceColor',[0 0.5 0],'MarkerEdgeColor',[0 0.5 0]);
            h.MarkerFaceAlpha = min(max(10/sum(ovf_zero),0.01),1);
            h.MarkerEdgeAlpha = min(max(10/sum(ovf_zero),0.01),1);
        end

        % Color the chkpos bit's overflow rates if overflow/underflow exists
        if ii == chkpos_idx && ovf_percent_zero(ii) > 0
            text(ii, -0.05, [num2str(ovf_percent_zero(ii),'%.1f'),'%'], 'Color', [0 0.5 0], 'FontWeight', 'bold');
        else
            text(ii, -0.05, [num2str(ovf_percent_zero(ii),'%.1f'),'%']);
        end
        if ii == chkpos_idx && ovf_percent_one(ii) > 0
            text(ii, 1.05, [num2str(ovf_percent_one(ii),'%.1f'),'%'], 'Color', 'r', 'FontWeight', 'bold');
        else
            text(ii, 1.05, [num2str(ovf_percent_one(ii),'%.1f'),'%']);
        end
    end

    % Add text notes to explain overflow rate meanings
    text(0, 1.05, 'Overflow rate:', 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
    text(0, -0.05, 'Underflow rate:', 'FontWeight', 'bold', 'HorizontalAlignment', 'right');

    axis([0,M+1,-0.1,1.1]);
    xticks(1:M);
    xticklabels(M:-1:1);
    xlabel('bit');
    ylabel('Residue Distribution');

    % Add legend
    legend([hBoundary, hMinMax, hAvg, hNormal, hOverflow, hUnderflow], ...
           {'Boundary (0/1)', 'Min/Max Range', 'Bit Average', 'Normal Samples', 'Overflow (>=1)', 'Underflow (<=0)'}, ...
           'Location', 'bestoutside');
end

end