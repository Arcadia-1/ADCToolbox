function [k2, k3, fitted_sine, fitted_transfer] = fitstaticnl(sig, order, freq)
%FITSTATICNL Extract static nonlinearity coefficients from distorted sinewave
%   This function extracts 2nd-order (k2) and 3rd-order (k3) static nonlinearity
%   coefficients from a distorted single-tone signal. It CANNOT extract gain error
%   since the sine fitting absorbs amplitude into the reference.
%
%   Syntax:
%     [k2, k3] = FITSTATICNL(sig, order)
%     [k2, k3] = FITSTATICNL(sig, order, freq)
%     [k2, k3, fitted_sine, fitted_transfer] = FITSTATICNL(sig, order, freq)
%
%   Inputs:
%     sig - Distorted sinewave signal samples
%       Vector of real numbers
%     order - Polynomial order for fitting
%       Positive integer (>=2), typically 2-3
%       order=2: Quadratic nonlinearity only (k2)
%       order=3: Quadratic + cubic nonlinearity (k2, k3)
%     freq - Normalized input frequency (frequency/fs), optional
%       Scalar in range (0, 0.5)
%       If omitted or 0, frequency is automatically estimated
%       Default: 0 (auto-detect)
%
%   Outputs:
%     k2 - Quadratic nonlinearity coefficient
%       Scalar, represents 2nd-order distortion
%       For ideal ADC: k2 = 0
%       Returns NaN if order < 2
%     k3 - Cubic nonlinearity coefficient
%       Scalar, represents 3rd-order distortion
%       For ideal ADC: k3 = 0
%       Returns NaN if order < 3
%     fitted_sine - Fitted ideal sinewave input (reference signal)
%       Vector (N×1), same length as sig, in time order
%       This is the ideal sine wave extracted from the distorted signal
%     fitted_transfer - Fitted transfer curve for plotting, struct
%       struct with fields:
%         x: 1000 smooth input points from min to max (sorted)
%         y: polynomial-evaluated output at those points
%       For ideal system: y=x (straight line)
%
%   Transfer Function Model:
%     y = x + k2*x^2 + k3*x^3
%     where:
%       x = ideal input (zero-mean sine)
%       y = actual output (zero-mean)
%       k2, k3 = nonlinearity coefficients
%
%   Examples:
%     % Extract coefficients only
%     sig = 0.5*sin(2*pi*0.123*(0:999)') + distortion;
%     [k2, k3] = fitstaticnl(sig, 3);
%
%     % Full extraction with plotting
%     [k2, k3, fitted_sine, fitted_transfer] = fitstaticnl(sig, 3);
%     figure;
%     plot(fitted_transfer.x, fitted_transfer.y - fitted_transfer.x, 'r-', 'LineWidth', 2);
%     xlabel('Input (V)');
%     ylabel('Nonlinearity Error (V)');
%     title(sprintf('Static Nonlinearity: k2=%.4f, k3=%.4f', k2, k3));
%     grid on;
%
%   Notes:
%     - Input signal must contain predominantly a single-tone sinewave
%     - Gain error CANNOT be extracted from single-tone measurements
%     - The sine fitting absorbs amplitude variations into the reference
%     - For accurate results, signal should have good SNR (>40 dB)
%     - DC offset is automatically removed before fitting
%
%   Algorithm:
%     1. Fit ideal sinewave to input signal using sinfit
%     2. Extract zero-mean ideal input (x) and actual output (y)
%     3. Normalize x for numerical stability
%     4. Fit polynomial: y = polyfit(x_normalized, order)
%     5. Denormalize coefficients to get k2, k3 (normalized by k1)
%     6. Generate smooth transfer curve for plotting
%
%   See also: sinfit, errsin, polyfit, inlsin

    % Input validation
    if nargin < 2
        error('fitstaticnl:notEnoughInputs', ...
              'At least 2 arguments required: sig and order');
    end

    if nargin < 3
        freq = 0;  % Auto-detect frequency
    end

    if ~isnumeric(sig) || ~isreal(sig)
        error('fitstaticnl:invalidSignal', ...
              'Signal must be a real-valued numeric vector');
    end

    if ~isnumeric(order) || ~isscalar(order) || order < 2 || mod(order, 1) ~= 0
        error('fitstaticnl:invalidOrder', ...
              'Order must be an integer >= 2');
    end

    if order > 10
        warning('fitstaticnl:highOrder', ...
                'Polynomial order > 10 may cause numerical instability');
    end

    if ~isnumeric(freq) || ~isscalar(freq) || freq < 0 || freq >= 0.5
        error('fitstaticnl:invalidFreq', ...
              'Frequency must be a scalar in range [0, 0.5)');
    end

    % Ensure column vector orientation
    sig = sig(:);
    N = length(sig);

    if N < order + 2
        error('fitstaticnl:insufficientData', ...
              'Signal length (%d) must be > polynomial order (%d) + 1', N, order);
    end

    % Fit ideal sinewave to signal
    if freq == 0
        [fitted_sine, ~, ~, ~, ~] = sinfit(sig);
    else
        [fitted_sine, ~, ~, ~, ~] = sinfit(sig, freq);
    end

    % Extract transfer function components
    % x = ideal input (zero-mean)
    % y = actual output (zero-mean)
    x_ideal = fitted_sine - mean(fitted_sine);
    y_actual = sig - mean(sig);

    % Normalize for numerical stability
    % This prevents coefficient overflow for large amplitude signals
    x_max = max(abs(x_ideal));

    if x_max < 1e-10
        error('fitstaticnl:zeroSignal', ...
              'Signal amplitude too small for fitting (< 1e-10)');
    end

    x_norm = x_ideal / x_max;

    % Fit polynomial to transfer function
    % polycoeff: [c_n, c_(n-1), ..., c_1, c_0]
    polycoeff = polyfit(x_norm, y_actual, order);

    % Extract and denormalize coefficients
    % Transfer function: y = x + k2*x^2 + k3*x^3
    % After normalization: y = c1*(x/x_max) + c2*(x/x_max)^2 + ...
    % Therefore: k_i = c_i / (x_max^i)

    % Linear coefficient (we don't return this, but need it for normalization)
    k1 = polycoeff(end-1) / x_max;

    % Quadratic coefficient (k2) - normalized by k1 to represent pure 2nd-order distortion
    if order >= 2
        k2_abs = polycoeff(end-2) / (x_max^2);
        k2 = k2_abs / k1;  % Normalize to unity gain
    else
        k2 = NaN;
    end

    % Cubic coefficient (k3) - normalized by k1 to represent pure 3rd-order distortion
    if order >= 3
        k3_abs = polycoeff(end-3) / (x_max^3);
        k3 = k3_abs / k1;  % Normalize to unity gain
    else
        k3 = NaN;
    end

    % Calculate fitted transfer curve on smooth grid for plotting (1000 sorted points)
    if nargout >= 4
        % Create smooth x-axis from min to max of fitted sine
        x_smooth = linspace(min(fitted_sine), max(fitted_sine), 1000)';

        % Normalize smooth x values (same normalization as used in fitting)
        x_smooth_norm = (x_smooth - mean(fitted_sine)) / x_max;

        % Evaluate polynomial at smooth points
        y_smooth_norm = polyval(polycoeff, x_smooth_norm);

        % Convert back to original scale
        y_smooth = y_smooth_norm + mean(sig);

        % Return as struct with x and y fields (matches Python tuple structure)
        fitted_transfer.x = x_smooth;
        fitted_transfer.y = y_smooth;
    end

end
