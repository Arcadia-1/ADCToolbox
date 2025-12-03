function [range_min, range_max, ovf_percent_zero, ovf_percent_one] = ovfchk(bits, varargin)
%OVFCHK Check ADC overflow by analyzing bit segment residue distributions
%   This function analyzes ADC output data to detect overflow conditions in
%   bit segments. For each bit position, it calculates the normalized residue
%   (sub-code from that bit to LSB) and visualizes the distribution to identify
%   segments that reach their minimum (all 0) or maximum (all 1) limits, indicating
%   potential overflow.
%
%   Syntax:
%     OVFCHK(bits)
%     OVFCHK(bits, wgt)
%     OVFCHK(bits, wgt, chkpos)
%     [range_min, range_max, ovf_percent_zero, ovf_percent_one] = OVFCHK(...)
%   or using parameter pairs:
%     OVFCHK(bits, 'name', value)
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
%   Plot Description (when nargout == 0):
%     - X-axis: Bit position (MSB to LSB, labeled M to 1)
%     - Y-axis: Normalized residue distribution [0, 1]
%     - Blue dots: Normal samples (no overflow)
%     - Red dots: Samples with overflow (>= 1)
%     - Yellow dots: Samples with underflow (<= 0)
%     - Red lines: Min/max range of residue for each bit
%     - Black lines: Boundaries at 0 and 1
%     - Text: Percentage of samples at 0 (bottom) and 1 (top)
%
%   Examples:
%     % Check overflow for 10-bit ADC with default binary weights
%     bits = randi([0 1], 10000, 10);
%     ovfchk(bits)
%
%     % Check overflow with custom weights
%     wgt = 2.^(9:-1:0);
%     ovfchk(bits, wgt)
%
%     % Check overflow of the segment: from the 8th-bit to LSB
%     ovfchk(bits, wgt, 8)
%
%     % Get overflow statistics
%     [range_min, range_max, pct_zero, pct_one] = ovfchk(bits);
%
%   Notes:
%     - A bit segment is the sub-code formed from one bit to the LSB
%     - Residue is normalized by dividing by the sum of weights in the corresponding segment
%     - Function automatically transposes input if N < M
%     - Uses transparent markers to visualize density of data points
%     - Plot is only displayed when nargout == 0 (no output arguments)
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
parse(p, varargin{:});
wgt = p.Results.wgt;
chkpos = p.Results.chkpos;

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

% Only plot if no output arguments requested
if nargout == 0
    hold on;
    plot([0,M+1],[1,1],'-k');
    plot([0,M+1],[0,0],'-k');
    plot((1:M),range_min,'-r');
    plot((1:M),range_max,'-r');
    for ii = 1:M

        h = scatter(ones([1,sum(non_ovf)])*ii, data_decom(non_ovf,ii), 'MarkerFaceColor','b','MarkerEdgeColor','b');
        h.MarkerFaceAlpha = min(max(10/N,0.01),1);
        h.MarkerEdgeAlpha = min(max(10/N,0.01),1);

        h = scatter(ones([1,sum(ovf_one)])*ii-0.2, data_decom(ovf_one,ii), 'MarkerFaceColor','r','MarkerEdgeColor','r');
        h.MarkerFaceAlpha = min(max(10/N,0.01),1);
        h.MarkerEdgeAlpha = min(max(10/N,0.01),1);

        h = scatter(ones([1,sum(ovf_zero)])*ii+0.2, data_decom(ovf_zero,ii), 'MarkerFaceColor','y','MarkerEdgeColor','y');
        h.MarkerFaceAlpha = min(max(10/N,0.01),1);
        h.MarkerEdgeAlpha = min(max(10/N,0.01),1);

        text(ii, -0.05, [num2str(ovf_percent_zero(ii),'%.1f'),'%']);
        text(ii, 1.05, [num2str(ovf_percent_one(ii),'%.1f'),'%']);
    end

    axis([0,M+1,-0.1,1.1]);
    xticks(1:M);
    xticklabels(M:-1:1);
    xlabel('bit');
    ylabel('Residue Distribution');
end

end