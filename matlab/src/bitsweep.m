function [ENoB_sweep, nBits_vec] = bitsweep(bits, varargin)

[N, M] = size(bits);
if N < M, bits = bits'; [N, M] = size(bits); end

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

if freq == 0
    [~, ~, ~, ~, ~, freq] = wcalsine(bits, 'freq', 0, 'order', order);
end

ENoB_sweep = zeros(1, M);
nBits_vec = 1:M;

for nBits = 1:M
    bits_subset = bits(:, 1:nBits);

    try
        [~, ~, postCal_temp, ~, ~, ~] = wcalsine(bits_subset, 'freq', freq, 'order', order);
        [ENoB_temp, ~, ~, ~, ~, ~, ~, ~] = specPlot(postCal_temp, ...
            'label', 0, 'harmonic', harmonic, 'OSR', OSR, 'winType', winType);
        ENoB_sweep(nBits) = ENoB_temp;
    catch ME
        ENoB_sweep(nBits) = NaN;
        fprintf('FAILED: %s\n', ME.message);
    end
end

if doPlot
    plot(nBits_vec, ENoB_sweep, 'o-k', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    hold on; grid on;
    xlabel('Number of Bits Used for Calibration', 'FontSize', 16);
    ylabel('ENoB (bits)', 'FontSize', 16);
    title('ENoB vs Number of Bits Used for Calibration', 'FontSize', 16);
    xlim([0.5, M + 0.5]);
    xticks(1:M);
    set(gca, 'FontSize', 14);

    validENoB = ENoB_sweep(~isnan(ENoB_sweep));
    if ~isempty(validENoB)
        ylim([min(validENoB) - 0.5, max(validENoB) + 2]);
    end

    deltaENoB = [ENoB_sweep(1), diff(ENoB_sweep)];

    if ~isempty(validENoB)
        yOffset = (max(validENoB) - min(validENoB)) * 0.06;
    else
        yOffset = 0.1;
    end

    for i = 1:M
        if ~isnan(ENoB_sweep(i)) && ~isnan(deltaENoB(i))
            if i == 1
                annotationText = sprintf('%.2f', deltaENoB(i));
                textColor = [0, 0, 0];
            else
                annotationText = sprintf('+%.2f', deltaENoB(i));
                normalizedDelta = max(0, min(1, deltaENoB(i)));
                textColor = [1 - normalizedDelta, 0, 0];
            end

            text(nBits_vec(i), ENoB_sweep(i) + yOffset, annotationText, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                'FontSize', 12, 'FontWeight', 'bold', 'Color', textColor);
        end
    end

    maxENoB = max(ENoB_sweep(~isnan(ENoB_sweep)));
    if ~isempty(maxENoB)
        yline(maxENoB, '--r', sprintf('Max ENoB = %.2f', maxENoB), ...
            'LabelHorizontalAlignment', 'left');
    end

    hold off;
end
end
