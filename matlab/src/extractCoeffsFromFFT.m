function [k1, k2, k3, k4, k5] = extractCoeffsFromFFT(data, fin)
% extractCoeffsFromFFT - Extract transfer function coefficients from FFT
%
% For a polynomial transfer function: y = k1*x + k2*x^2 + k3*x^3 + k4*x^4 + k5*x^5
% Applied to x = A*sin(ωt), the harmonics are:
%   - Fundamental amplitude = k1*A + (3/4)*k3*A^3 + (5/8)*k5*A^5
%   - 2nd harmonic amplitude = (1/2)*k2*A^2 + (1/2)*k4*A^4
%   - 3rd harmonic amplitude = (1/4)*k3*A^3 + (5/16)*k5*A^5
%   - 4th harmonic amplitude = (1/8)*k4*A^4
%   - 5th harmonic amplitude = (1/16)*k5*A^5
%
% Inputs:
%   data - ADC output data
%   fin - Normalized frequency
%
% Outputs:
%   k1, k2, k3, k4, k5 - Transfer function coefficients

    N = length(data);

    % Get the ideal amplitude and phase from sine fit
    [~, ~, A, ~, ~] = sineFit(data, fin);

    % Compute FFT
    Y = fft(data);
    Y_mag = abs(Y) * 2 / N;  % Single-sided amplitude

    % Find bin for fundamental and harmonics
    J = round(fin * N);  % Fundamental bin

    % Extract harmonic amplitudes
    H1 = Y_mag(J + 1);      % Fundamental (MATLAB 1-indexed)
    H2 = 0; H3 = 0; H4 = 0; H5 = 0;

    if 2*J < N/2
        H2 = Y_mag(2*J + 1);  % 2nd harmonic
    end
    if 3*J < N/2
        H3 = Y_mag(3*J + 1);  % 3rd harmonic
    end
    if 4*J < N/2
        H4 = Y_mag(4*J + 1);  % 4th harmonic
    end
    if 5*J < N/2
        H5 = Y_mag(5*J + 1);  % 5th harmonic
    end

    % Extract coefficients from harmonics
    % Start from highest order and work backwards

    % From 5th harmonic: H5 = (1/16)*k5*A^5
    if H5 > 1e-6
        k5 = H5 / (A^5 / 16);
    else
        k5 = 0;
    end

    % From 4th harmonic: H4 = (1/8)*k4*A^4
    if H4 > 1e-6
        k4 = H4 / (A^4 / 8);
    else
        k4 = 0;
    end

    % From 3rd harmonic: H3 = (1/4)*k3*A^3 + (5/16)*k5*A^5
    if H3 > 1e-6
        k3 = (H3 - (5/16)*k5*A^5) / (A^3 / 4);
    else
        k3 = 0;
    end

    % From 2nd harmonic: H2 = (1/2)*k2*A^2 + (1/2)*k4*A^4
    if H2 > 1e-6
        k2 = (H2 - (1/2)*k4*A^4) / (A^2 / 2);
    else
        k2 = 0;
    end

    % From fundamental: H1 = k1*A + (3/4)*k3*A^3 + (5/8)*k5*A^5
    k1 = (H1 - (3/4)*k3*A^3 - (5/8)*k5*A^5) / A;

end
