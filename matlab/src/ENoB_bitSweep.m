function [ENoB_sweep, nBits_vec] = ENoB_bitSweep(bits, varargin)
% ENoB_bitSweep - Sweep ENoB vs number of bits used for calibration
%
% This function evaluates the Effective Number of Bits (ENoB) as a function
% of the number of bits used for foreground calibration with FGCalSine.
% It helps determine the optimal number of bits needed for calibration.
%
% Usage:
%   [ENoB_sweep, nBits_vec] = ENoB_bitSweep(bits, [Name, Value])
%
% Inputs:
%   - bits: binary data matrix (N samples × M bits)
%     Each row is one sample, each column is a bit
%
% Name-Value Arguments:
%   - freq: normalized frequency Fin/Fs (0 triggers search), default is 0
%   - order: harmonics exclusion order for FGCalSine, default is 5
%   - harmonic: number of harmonics for specPlot, default is 5
%   - OSR: oversampling ratio for specPlot, default is 1
%   - winType: window function for specPlot, default is @hamming
%   - plot: whether to create plot (1) or not (0), default is 1
%
% Outputs:
%   - ENoB_sweep: vector of ENoB values (1 × M)
%   - nBits_vec: vector of number of bits used (1 × M)
%
% Example:
%   [ENoB_sweep, nBits] = ENoB_bitSweep(bits, 'freq', 0.1234, 'plot', 1);
%
% See also: FGCalSine, specPlot

% Parse inputs
[N, M] = size(bits);
if N < M
    bits = bits';
    [N, M] = size(bits);
end

p = inputParser;
addOptional(p, 'freq', 0, @(x) isnumeric(x) && isscalar(x) && (x >= 0));
addOptional(p, 'order', 5, @(x) isnumeric(x) && isscalar(x) && (x > 0));
addOptional(p, 'harmonic', 5, @(x) isnumeric(x) && isscalar(x) && (x > 0));
addOptional(p, 'OSR', 1, @(x) isnumeric(x) && isscalar(x) && (x > 0));
addOptional(p, 'winType', @hamming, @(x) isa(x, 'function_handle'));
addOptional(p, 'plot', 1, @(x) isnumeric(x) && isscalar(x));
parse(p, varargin{:});

freq = p.Results.freq;
order = p.Results.order;
harmonic = p.Results.harmonic;
OSR = p.Results.OSR;
winType = p.Results.winType;
doPlot = p.Results.plot;

% First pass: determine frequency if not provided
if freq == 0
    fprintf('ENoB_bitSweep: Determining frequency using all bits...\n');
    [~, ~, ~, ~, ~, freq] = FGCalSine(bits, 'freq', 0, 'order', order);
    fprintf('  Frequency found: %.6f\n', freq);
end

% Initialize output
ENoB_sweep = zeros(1, M);
nBits_vec = 1:M;

% Sweep through different numbers of bits
for nBits = 1:M
    fprintf('  [%d/%d] Testing with %d bits... ', nBits, M, nBits);

    % Use only first nBits for calibration
    bits_subset = bits(:, 1:nBits);

    try
        % Run FGCalSine with fixed frequency
        [~, ~, postCal_temp, ~, ~, ~] = FGCalSine(bits_subset, 'freq', freq, 'order', order);

        % Compute ENoB using specPlot
        [ENoB_temp, ~, ~, ~, ~, ~, ~, ~] = specPlot(postCal_temp, ...
            'label', 0, 'harmonic', harmonic, 'OSR', OSR, 'winType', winType);

        ENoB_sweep(nBits) = ENoB_temp;
        fprintf('ENoB = %.2f bits\n', ENoB_temp);

    catch ME
        ENoB_sweep(nBits) = NaN;
        fprintf('FAILED: %s\n', ME.message);
    end
end

% Create plot if requested
if doPlot
    plot(nBits_vec, ENoB_sweep, 'o-k', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    hold on;
    grid on;
    xlabel('Number of Bits Used for Calibration', 'FontSize', 16);
    ylabel('ENoB (bits)', 'FontSize', 16);
    title('ENoB vs Number of Bits Used for Calibration', 'FontSize', 16);
    xlim([0.5, M + 0.5]);
    xticks(1:M);

    % Set font size for axes tick labels
    set(gca, 'FontSize', 14);

    % Set y-axis limits to accommodate annotations
    validENoB = ENoB_sweep(~isnan(ENoB_sweep));
    if ~isempty(validENoB)
        maxENoB_data = max(validENoB);
        minENoB_data = min(validENoB);
        ylim([minENoB_data - 0.5, maxENoB_data + 2]); % Max + 1 as requested
    end

    % Calculate and annotate incremental ENoB contribution of each bit
    % deltaENoB(1) = absolute ENoB of first bit
    % deltaENoB(i) = ENoB improvement when adding bit i (i > 1)
    deltaENoB = [ENoB_sweep(1), diff(ENoB_sweep)]; % First bit = absolute, rest = delta

    % Determine y-axis range for annotation positioning
    if ~isempty(validENoB)
        yRange = max(validENoB) - min(validENoB);
        yOffset = yRange * 0.06; % 6% of range for vertical offset above points
    else
        yOffset = 0.1;
    end

    % Annotate each point with delta ENoB
    for i = 1:M
        if ~isnan(ENoB_sweep(i)) && ~isnan(deltaENoB(i))
            % Format annotation text
            if i == 1
                % First bit shows absolute ENoB (no sign)
                annotationText = sprintf('%.2f', deltaENoB(i));
                textColor = [0, 0, 0]; % Black for first bit
            else
                % Subsequent bits show delta with + sign
                annotationText = sprintf('+%.2f', deltaENoB(i));

                % Color coding based on ABSOLUTE delta value (fixed 0-1 scale)
                % This provides consistent color meaning across different datasets:
                %   Delta ≥ 1.0 → Black (adds ≥1 effective bit, critical)
                %   Delta ≈ 0.5 → Gray (adds ~0.5 bits, moderate)
                %   Delta ≤ 0.0 → Red (minimal/harmful)
                normalizedDelta = deltaENoB(i); % Use absolute delta value
                normalizedDelta = max(0, min(1, normalizedDelta)); % Clamp to [0, 1]

                % Create red-to-black color gradient
                % normalizedDelta = 0 → Red [1, 0, 0] (no improvement)
                % normalizedDelta = 0.5 → Dark orange [0.5, 0, 0] (half a bit)
                % normalizedDelta = 1 → Black [0, 0, 0] (one full bit)
                redComponent = 1 - normalizedDelta; % Decreases from 1 to 0
                textColor = [redComponent, 0, 0];
            end

            % Position annotation above point
            yPos = ENoB_sweep(i) + yOffset;

            % Add text annotation
            text(nBits_vec(i), yPos, annotationText, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontSize', 12, ...
                'FontWeight', 'bold', ...
                'Color', textColor);
        end
    end

    % Add horizontal line at maximum ENoB
    maxENoB = max(ENoB_sweep(~isnan(ENoB_sweep)));
    if ~isempty(maxENoB)
        yline(maxENoB, '--r', sprintf('Max ENoB = %.2f', maxENoB), ...
            'LabelHorizontalAlignment', 'left');
    end

    hold off;
end

fprintf('ENoB_bitSweep: Complete\n');

end
