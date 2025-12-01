function [inl, dnl, code] = inlsin(data, varargin)
%INLSIN Calculate ADC's INL and DNL from sinewave data by histogram method
%   This function computes Integral Nonlinearity (INL) and Differential
%   Nonlinearity (DNL) of an ADC using the histogram method.
%   The method assumes the input is a pure sinewave and uses the
%   cumulative histogram to reconstruct the transfer curve.
%
%   Syntax:
%     [inl, dnl, code] = INLSIN(data)
%     [inl, dnl, code] = INLSIN(data, excl)
%     [inl, dnl, code] = INLSIN(data, excl, disp)
%   or using parameter pairs:
%     [inl, dnl, code] = INLSIN(data, 'name', value, ...)
%
%   Inputs:
%     data - ADC output codes from sinewave input
%       Vector (integer values, will be rounded if non-integer with warning)
%       IMPORTANT: Data should be integer ADC codes
%     excl - Exclusion ratio for endpoints (optional, default: 0.01)
%       Scalar in range [0, 0.5)
%       Excludes excl*100% of codes from each end to avoid noise issue
%     disp - Display switch (optional)
%       Logical
%       Default: nargout == 0 (auto-display when no outputs)
%
%   Outputs:
%     inl - Integral Nonlinearity in LSB
%       Vector (same length as code)
%       Cumulative sum of DNL errors
%     dnl - Differential Nonlinearity in LSB
%       Vector (same length as code)
%       Deviation from ideal 1 LSB code width
%     code - Code values corresponding to INL/DNL measurements
%       Vector
%
%   Examples:
%     % Generate ADC output from sine wave and calculate INL/DNL
%     t = linspace(0, 2*pi, 10000);
%     data = round(127.5 + 127.5*sin(t));  % 8-bit ADC
%     [inl, dnl, code] = inlsin(data);
%
%     % Custom exclusion ratio to exclude more endpoints
%     [inl, dnl, code] = inlsin(data, 0.05);  % Exclude 5% from each end
%
%     % Force display with outputs
%     [inl, dnl, code] = inlsin(data, 0.01, true);
%
%     % Auto-display when no outputs
%     inlsin(data)
%
%   Notes:
%     - Input data should be integer ADC codes (non-integer will be rounded)
%     - This function assumes LSB = 1 (unit code width)
%     - The sine histogram method assumes an ideal sinewave input
%     - DNL is normalized to have zero mean
%     - Missing codes (DNL <= -1) are highlighted in red when displayed
%
%   Algorithm:
%     1. Histogram the ADC output codes
%     2. Apply cos transform to linearize sine distribution
%     3. Calculate DNL from differences in linearized histogram
%     4. Calculate INL as cumulative sum of DNL
%
%   See also: histcounts, cumsum

    % Parse input arguments
    p = inputParser;
    addOptional(p, 'excl', 0.01, @(x) isnumeric(x) && isscalar(x) && (x >= 0) && (x < 0.5));
    addOptional(p, 'disp', nargout == 0, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    parse(p, varargin{:});
    excl = p.Results.excl;
    disp_flag = logical(p.Results.disp);

    % Input validation
    if ~isnumeric(data) || ~isreal(data)
        error('inlsin:invalidData', 'Input data must be real numeric values.');
    end

    % Check if data is integer (use tolerance to avoid floating-point precision issues)
    if any(abs(data(:) - round(data(:))) > 2*eps)
        warning('inlsin:nonIntegerData', 'Input data contains non-integer values. Rounding to nearest integer.');
        data = round(data);
    end

    % Ensure data is a column vector
    S = size(data);
    if(S(1) < S(2))
        data = data';
    end

    % Determine code range and apply initial exclusion to avoid clipping
    max_data_orig = max(data);
    min_data_orig = min(data);

    % Calculate exclusion amount (fix: store original values before modification)
    exclusion_amount = round(excl * (max_data_orig - min_data_orig));
    max_data = max_data_orig - exclusion_amount;
    min_data = min(min_data_orig + exclusion_amount, max_data);

    % Create code range and clip data to valid range
    code = min_data:max_data;
    data = min(max(data, min_data), max_data);

    % Compute histogram and apply cosine transform
    % This linearizes the sine wave distribution
    histogram_counts = histcounts(data, [code-0.5, code(end)+0.5]);
    cumulative_distribution = -cos(pi * cumsum(histogram_counts) / sum(histogram_counts));

    % Calculate DNL from differences in linearized distribution
    dnl = cumulative_distribution(2:end-1) - cumulative_distribution(1:end-2);
    code = code(1:end-2);

%    % Apply secondary exclusion to remove edge effects
%    excl_codes = floor(excl * (max_data - min_data + 1) / 2);
%    code = code(excl_codes+1 : end-excl_codes);
%    dnl = dnl(excl_codes+1 : end-excl_codes);

    % Normalize DNL to LSB units
    dnl = dnl ./ sum(dnl);
    dnl = dnl * (max_data - min_data + 1) - 1;
    dnl = dnl - mean(dnl);  % Remove DC offset
    dnl = max(dnl, -1);

    % Calculate INL as cumulative sum of DNL
    inl = cumsum(dnl);

    % Display results if requested
    if disp_flag
        subplot(2,1,1);
        plot(code, dnl, 'k-');
        missing = (dnl <= -1);
        if(sum(missing) > 0)
            hold on;
            plot(code(missing), dnl(missing), 'ro');
            legend('DNL', 'Missing Code');
        end
        grid on;
        xlim([min_data_orig, max_data_orig]);
        xlabel('Code');
        ylabel('DNL (LSB)');
        title('Differential Nonlinearity (DNL)');


        subplot(2,1,2);
        plot(code, inl, 'k-');
        grid on;
        xlim([min_data_orig, max_data_orig]);
        xlabel('Code');
        ylabel('INL (LSB)');
        title('Integral Nonlinearity (INL)');
    end

end
