function [k0_exp, k1_exp, k2_exp, k3_exp, k4_exp, k5_exp] = calcExpectedCoeffs(filename)
% calcExpectedCoeffs - Calculate expected nonlinearity coefficients from filename
%
% Based on generate_sinewave_harmonics.m:
%   - Amplitude A = 0.499
%   - coef2 = hd2_amp / (A/2) where hd2_amp = 10^(HD2_dB/20)
%   - coef3 = hd3_amp / (A^2/4) where hd3_amp = 10^(HD3_dB/20)
%   - Transfer function: y = x + coef2*x^2 + coef3*x^3 + ... + 0.5 (DC)
%
% Returns:
%   k0_exp - Expected offset (should be 0.5 from generation)
%   k1_exp - Expected gain (should be 1.0)
%   k2_exp - Expected 2nd-order nonlinearity (coef2)
%   k3_exp - Expected 3rd-order nonlinearity (coef3)
%   k4_exp - Expected 4th-order nonlinearity (coef4) or NaN
%   k5_exp - Expected 5th-order nonlinearity (coef5) or NaN

    A = 0.499;  % Amplitude from generation script

    % Initialize all to NaN
    k0_exp = 0.5;  % DC offset
    k1_exp = 1.0;  % Ideal gain
    k2_exp = NaN;
    k3_exp = NaN;
    k4_exp = NaN;
    k5_exp = NaN;

    % Parse HD2 from filename
    hd2_match = regexp(filename, 'HD2_n(\d+)dB', 'tokens');
    if ~isempty(hd2_match)
        HD2_dB = -str2double(hd2_match{1}{1});
        hd2_amp = 10^(HD2_dB / 20);
        k2_exp = hd2_amp / (A / 2);  % coef2
    end

    % Parse HD3 from filename
    hd3_match = regexp(filename, 'HD3_n(\d+)dB', 'tokens');
    if ~isempty(hd3_match)
        HD3_dB = -str2double(hd3_match{1}{1});
        hd3_amp = 10^(HD3_dB / 20);
        k3_exp = hd3_amp / (A^2 / 4);  % coef3
    end

    % Parse HD4 from filename
    hd4_match = regexp(filename, 'HD4_n(\d+)dB', 'tokens');
    if ~isempty(hd4_match)
        HD4_dB = -str2double(hd4_match{1}{1});
        hd4_amp = 10^(HD4_dB / 20);
        k4_exp = hd4_amp / (A^3 / 8);  % coef4
    end

    % Parse HD5 from filename
    hd5_match = regexp(filename, 'HD5_n(\d+)dB', 'tokens');
    if ~isempty(hd5_match)
        HD5_dB = -str2double(hd5_match{1}{1});
        hd5_amp = 10^(HD5_dB / 20);
        k5_exp = hd5_amp / (A^4 / 16);  % coef5
    end
end
