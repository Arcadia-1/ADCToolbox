function [k1, k2, k3, polycoeff, fit_curve] = fitstaticnl(sig, order, freq)
%FITSTATICNL Extract static nonlinearity coefficients from ADC transfer function
%   This function fits a polynomial to the ADC transfer function to extract
%   static nonlinearity coefficients. It models the relationship between
%   ideal input (fitted sine wave) and actual output (measured signal).
%
%   Syntax:
%     [k1, k2, k3] = FITSTATICNL(sig, order)
%     [k1, k2, k3] = FITSTATICNL(sig, order, freq)
%     [k1, k2, k3, polycoeff, fit_curve] = FITSTATICNL(sig, order, freq)
%
%   Inputs:
%     sig - ADC output signal (sinewave samples)
%       Vector of real numbers
%     order - Polynomial order for fitting
%       Positive integer (1-10), typically 2-4
%       order=1: Linear gain only (k1)
%       order=2: Linear + quadratic (k1, k2)
%       order=3: Linear + quadratic + cubic (k1, k2, k3)
%     freq - Normalized input frequency (frequency/fs), optional
%       Scalar in range (0, 0.5)
%       If omitted or 0, frequency is automatically estimated
%       Default: 0 (auto-detect)
%
%   Outputs:
%     k1 - Linear gain coefficient
%       Scalar, represents ideal linear gain
%       For ideal ADC: k1 = 1.0
%     k2 - Quadratic nonlinearity coefficient
%       Scalar, represents 2nd-order distortion
%       For ideal ADC: k2 = 0
%       Returns NaN if order < 2
%     k3 - Cubic nonlinearity coefficient
%       Scalar, represents 3rd-order distortion
%       For ideal ADC: k3 = 0
%       Returns NaN if order < 3
%     polycoeff - Full polynomial coefficients (highest to lowest order)
%       Vector [c_n, c_(n-1), ..., c_1, c_0]
%       Transfer function: y = c_n*x^n + ... + c_1*x + c_0
%     fit_curve - Fitted transfer function evaluated at signal points
%       Vector (N×1), same length as sig
%       Useful for plotting fitted curve
%
%   Transfer Function Model:
%     y = k1*x + k2*x^2 + k3*x^3 + ...
%     where:
%       x = ideal input (zero-mean)
%       y = actual output (zero-mean)
%
%   Examples:
%     % Extract linear and quadratic coefficients (order=2)
%     sig = 0.5*sin(2*pi*0.123*(0:999)') + 0.01*randn(1000,1);
%     [k1, k2] = fitstaticnl(sig, 2)
%
%     % Extract up to cubic nonlinearity with auto frequency detection
%     [k1, k2, k3] = fitstaticnl(sig, 3)
%
%     % Specify frequency explicitly for faster computation
%     [k1, k2, k3] = fitstaticnl(sig, 3, 0.123)
%
%     % Get full polynomial and plot transfer function
%     [k1, k2, k3, polycoeff, fit_curve] = fitstaticnl(sig, 3);
%     [sig_fit, ~, ~, ~, ~] = sinfit(sig);
%     figure;
%     plot(sig_fit, sig, 'b.', sig_fit, fit_curve, 'r-', 'LineWidth', 2);
%     xlabel('Ideal Input'); ylabel('Actual Output');
%     legend('Measured', 'Fitted');
%     title(sprintf('Transfer Function: k1=%.4f, k2=%.4f, k3=%.4f', k1, k2, k3));
%
%   Notes:
%     - Input signal must contain predominantly a single-tone sinewave
%     - Coefficients are normalized (k1 ≈ 1.0 for ideal ADC)
%     - Higher-order terms (k2, k3, ...) represent static distortion
%     - For accurate results, signal should have good SNR (>40 dB)
%     - Coefficients are denormalized to handle any amplitude range
%     - DC offset is automatically removed before fitting
%
%   Algorithm:
%     1. Fit ideal sinewave to input signal using sinfit
%     2. Extract zero-mean ideal input (x) and actual output (y)
%     3. Normalize x for numerical stability
%     4. Fit polynomial: y = polyfit(x_normalized, order)
%     5. Denormalize coefficients to get physical k1, k2, k3
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

    if ~isnumeric(order) || ~isscalar(order) || order < 1 || mod(order, 1) ~= 0
        error('fitstaticnl:invalidOrder', ...
              'Order must be a positive integer');
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
        [sig_fit, ~, ~, ~, ~] = sinfit(sig);
    else
        [sig_fit, ~, ~, ~, ~] = sinfit(sig, freq);
    end

    % Extract transfer function components
    % x = ideal input (zero-mean)
    % y = actual output (zero-mean)
    x_ideal = sig_fit - mean(sig_fit);
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
    % Transfer function: y = k1*x + k2*x^2 + k3*x^3 + ...
    % After normalization: y = c1*(x/x_max) + c2*(x/x_max)^2 + ...
    % Therefore: k_i = c_i / (x_max^i)

    % Linear coefficient (k1)
    k1 = polycoeff(end-1) / x_max;

    % Quadratic coefficient (k2)
    if order >= 2
        k2 = polycoeff(end-2) / (x_max^2);
    else
        k2 = NaN;
    end

    % Cubic coefficient (k3)
    if order >= 3
        k3 = polycoeff(end-3) / (x_max^3);
    else
        k3 = NaN;
    end

    % Calculate fitted curve if requested
    if nargout >= 5
        % Evaluate polynomial at normalized input points
        y_fit_norm = polyval(polycoeff, x_norm);

        % Convert back to original scale (add mean back)
        fit_curve = y_fit_norm + mean(sig);
    end

end
