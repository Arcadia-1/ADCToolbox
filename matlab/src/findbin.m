function b = findbin(fs, fin, n)
%FINDBIN Find coherent FFT bin for a given signal frequency
%   This function calculates the FFT bin index for a signal frequency that
%   ensures coherent sampling (integer number of cycles in the FFT window).
%   It finds the nearest bin where the bin number and FFT size are coprime
%   (gcd = 1), which ensures the signal completes an integer number of cycles,
%   and the sampled phases are not repeated.
%
%   Syntax:
%     b = FINDBIN(fs, fin, n)
%
%   Inputs:
%     fs - Sampling frequency
%       Scalar positive real number
%     fin - Input signal frequency
%       Scalar or vector real numbers
%     n - FFT size (number of samples)
%       Scalar positive integer
%
%   Outputs:
%     b - the nearest FFT bin index for coherent sampling
%       Scalar or vector of positive integers (same size as fin)
%       Satisfies: gcd(b, n) = 1 (coprime condition)
%
%   Examples:
%     % Find coherent bin for 1 kHz signal at 10 kHz sampling, 1024 samples
%     b = findbin(10000, 1000, 1024)
%     % Returns b = 103
%
%     % Find the nearest coherent signal frequency under given fs and n
%     fs = 100e3; n = 8192;
%     fin_desired = 12.5e3;
%     b = findbin(fs, fin_desired, n)     % = 1025
%     fin_actual = b * fs / n             % = 12.512e3
%
%     % Process multiple frequencies at once
%     fin_vec = [1000, 2500, 3750];
%     b_vec = findbin(10000, fin_vec, 1024)
%     % Returns b_vec = [103, 257, 385] 
%
%   Notes:
%     - The function searches both upward and downward from the initial bin
%       estimate to find the coherent bin closest to the desired frequency
%     - If two bins are equidistant, the upper (larger) bin is returned
%     - The returned bin is always greater than 0
%
%   See also: FFT, GCD

    % Input validation
    if nargin < 3
        error('findbin:notEnoughInputs', ...
              'Three input arguments required: fs, fin, n.');
    end

    if ~isvector(fin)
        error('findbin:invalidInput', ...
              'fin must be a vector.');
    end

    if ~isscalar(fs) || ~isscalar(n)
        error('findbin:invalidInput', ...
              'fs and n must be scalars.');
    end

    if fs <= 0 
        error('findbin:invalidFrequency', ...
              'fs must be positive.');
    end

    if n <= 0 || floor(n) ~= n
        error('findbin:invalidN', ...
              'FFT size n must be a positive integer.');
    end

    if ~isreal(fs) || ~isreal(fin) || ~isreal(n)
        error('findbin:invalidInput', ...
              'All inputs must be real numbers.');
    end

    % Preallocate output array
    b = zeros(size(fin));

    % Process each frequency in fin
    for i = 1:length(fin)
        % Calculate initial bin estimate
        bin_start = floor(fin(i) / fs * n);

        % Search for nearest coherent bin by incrementing distance
        d = 0;
        while true
            % Check upper bin first (bin_start + d)
            b_upper = bin_start + d;
            if gcd(b_upper, n) == 1
                b(i) = b_upper;
                break;
            end

            % Check lower bin (bin_start - d) if valid
            b_lower = bin_start - d;
            if b_lower > 0 && gcd(b_lower, n) == 1
                b(i) = b_lower;
                break;
            end

            % Increment distance
            d = d + 1;
        end
    end

end
