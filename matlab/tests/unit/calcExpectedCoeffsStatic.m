function [k1_exp, k2_exp, k3_exp, k4_exp, k5_exp] = calcExpectedCoeffsStatic(filename)
% calcExpectedCoeffsStatic - Parse static nonlinearity coefficients from filename
%
% For static nonlinearity files: static_nonlin_k1_1.000_k2_0.050_k3_0.100_N_4096.csv
%
% Returns:
%   k1_exp, k2_exp, k3_exp, k4_exp, k5_exp - Expected transfer function coefficients

    % Initialize all to NaN
    k1_exp = NaN;
    k2_exp = NaN;
    k3_exp = NaN;
    k4_exp = NaN;
    k5_exp = NaN;

    % Parse k1 from filename
    k1_match = regexp(filename, 'k1_([\d\.]+)', 'tokens');
    if ~isempty(k1_match)
        k1_exp = str2double(k1_match{1}{1});
    end

    % Parse k2 from filename
    k2_match = regexp(filename, 'k2_([\d\.]+)', 'tokens');
    if ~isempty(k2_match)
        k2_exp = str2double(k2_match{1}{1});
    end

    % Parse k3 from filename
    k3_match = regexp(filename, 'k3_([\d\.]+)', 'tokens');
    if ~isempty(k3_match)
        k3_exp = str2double(k3_match{1}{1});
    end

    % Parse k4 from filename
    k4_match = regexp(filename, 'k4_([\d\.]+)', 'tokens');
    if ~isempty(k4_match)
        k4_exp = str2double(k4_match{1}{1});
    end

    % Parse k5 from filename
    k5_match = regexp(filename, 'k5_([\d\.]+)', 'tokens');
    if ~isempty(k5_match)
        k5_exp = str2double(k5_match{1}{1});
    end
end
