function [k0, k1, k2, k3, k4, k5, x_ideal, y_actual, polycoeff] = extractTransferFunction(data, fin, polyorder)
% extractTransferFunction - Extract transfer function coefficients from ADC data
%
% This function extracts the actual transfer function: y = f(x)
% where x is the ideal input and y is the actual ADC output
%
% Inputs:
%   data - ADC output data (sine wave with distortion)
%   fin - Normalized frequency (0 to 1)
%   polyorder - Polynomial order (e.g., 3 for up to x^3)
%
% Outputs:
%   k0, k1, k2, k3, k4, k5 - Transfer function coefficients
%   x_ideal - Ideal input values (sine wave)
%   y_actual - Actual output values (with DC removed)
%   polycoeff - Raw polynomial coefficients

    % Fit ideal sine wave to get what the input should have been
    if fin == 0
        [data_fit, fin, mag, ~, phi] = sineFit(data);
    else
        [data_fit, ~, mag, ~, phi] = sineFit(data, fin);
    end

    % The ideal input is the fitted sine wave (zero mean)
    x_ideal = data_fit - mean(data_fit);

    % The actual output with DC removed
    y_actual = data - mean(data);

    % Now fit: y_actual = k0 + k1*x_ideal + k2*x_ideal^2 + k3*x_ideal^3 + ...
    % Normalize x to [-1, 1] for numerical stability
    x_max = max(abs(x_ideal));
    x_norm = x_ideal / x_max;

    % Fit polynomial
    polycoeff = polyfit(x_norm, y_actual, polyorder);

    % Extract coefficients
    % y = p_n*x^n + ... + p_1*x + p_0
    k0 = polycoeff(end);

    % Need to denormalize coefficients
    % If y = sum(p_i * (x/x_max)^i), then y = sum(p_i * x^i / x_max^i)
    % So actual k_i = p_i / x_max^i

    k1 = polycoeff(end-1) / x_max;

    if polyorder >= 2
        k2 = polycoeff(end-2) / (x_max^2);
    else
        k2 = 0;
    end

    if polyorder >= 3
        k3 = polycoeff(end-3) / (x_max^3);
    else
        k3 = 0;
    end

    if polyorder >= 4
        k4 = polycoeff(end-4) / (x_max^4);
    else
        k4 = 0;
    end

    if polyorder >= 5
        k5 = polycoeff(end-5) / (x_max^5);
    else
        k5 = 0;
    end

end
