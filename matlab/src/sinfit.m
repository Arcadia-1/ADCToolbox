function [fitout,freq,mag,dc,phi] = sinfit(sig,varargin)
%SINFIT Four-parameter iterative sine wave fitting
%   This function performs a 4-parameter sine wave fit to input signal using
%   an iterative least-squares method. The four parameters are: amplitude,
%   phase, DC offset, and frequency. The frequency is refined iteratively
%   using a gradient descent approach until convergence.
%
%   Syntax:
%     [fitout, freq, mag, dc, phi] = SINFIT(sig)
%     [fitout, freq, mag, dc, phi] = SINFIT(sig, f0)
%     [fitout, freq, mag, dc, phi] = SINFIT(sig, f0, tol)
%     [fitout, freq, mag, dc, phi] = SINFIT(sig, f0, tol, rate)
%     [fitout, freq, mag, dc, phi] = SINFIT(sig, f0, tol, rate, fsearch)
%     [fitout, freq, mag, dc, phi] = SINFIT(sig, f0, tol, rate, fsearch, verbose)
%     [fitout, freq, mag, dc, phi] = SINFIT(sig, 'Name', Value, ...)
%
%   Inputs:
%     sig - Input signal to be fitted
%       Vector (row or column) or Matrix (averaged across columns)
%
%   Name-Value Arguments (or positional in order: f0, tol, rate, fsearch, verbose):
%     f0 - Initial frequency estimate (normalized by sample count)
%       Scalar, Range: [0, 0.5]
%       Default: 0 (triggers automatic estimation from FFT peak)
%     tol - Convergence tolerance for relative error
%       Scalar, positive real number
%       Default: 1e-12
%     rate - Step size rate for frequency update (learning rate)
%       Scalar, Range: (0, 1]
%       Default: 0.5
%     fsearch - Force fine frequency search iteration
%       Scalar, {0, 1}
%       Default: 0 (auto-enabled when f0=0)
%       Set to 1 to enable iterative frequency refinement
%     verbose - Enable verbose output during iteration
%       Scalar, {0, 1}
%       Default: 0
%       Set to 1 to print iteration progress messages
%
%   Outputs:
%     fitout - Fitted sine wave signal
%       Vector (column)
%     freq - Fitted normalized frequency (cycles per sample)
%       Scalar, Range: [0, 0.5]
%     mag - Fitted signal amplitude (peak value)
%       Scalar, non-negative
%     dc - Fitted DC offset
%       Scalar
%     phi - Fitted phase in radians
%       Scalar, Range: [-pi, pi]
%
%     Convention: fitout = mag*cos(2*pi*freq*t + phi) + dc
%
%   Examples:
%     % Fit a noisy sine wave with known frequency
%     t = 0:99;
%     sig = 3*cos(2*pi*0.1*t - pi/4) + 0.5 + 0.1*randn(1,100);
%     [fitout, freq, mag, dc, phi] = sinfit(sig, 0.1);
%
%   Algorithm:
%     1. Initial 3-parameter fit (A, B, dc) using linear least squares
%        with cos/sin basis at estimated frequency
%     2. Iterative refinement: compute gradient of frequency error,
%        update frequency, and re-solve least squares
%     3. Converge when relative frequency error < tol or 100 iterations
%     4. Convert to amplitude-phase form: mag = sqrt(A^2+B^2), phi = atan2(B,A)
%
%   Notes:
%     - For matrix input, signal is averaged across columns first
%     - Maximum 100 iterations
%     - Frequency f0 is normalized: f0 = f_Hz / f_sample
%     - Phase convention: positive phase = signal leads cos(2*pi*freq*t)
%
%   See also: fft, lsqcurvefit, nlinfit

    % Input validation
    if ~isnumeric(sig) || ~isreal(sig)
        error('sinfit:invalidInput', 'Input signal must be a real numeric array.');
    end

    if isempty(sig)
        error('sinfit:emptyInput', 'Input signal cannot be empty.');
    end

    % Reshape input to column vector and average across columns
    [N,M] = size(sig);
    if(N == 1)
       sig = sig';
       N = M;
    end
    sig = mean(sig,2);

    % Parse optional inputs (order: f0, tol, rate, fsearch, verbose for backward compatibility)
    p = inputParser;
    addOptional(p, 'f0', 0, @(x) isnumeric(x) && isscalar(x) && (x >= 0) && (x <= 0.5));
    addOptional(p, 'tol', 1e-12, @(x) isnumeric(x) && isscalar(x) && (x > 0));
    addOptional(p, 'rate', 0.5, @(x) isnumeric(x) && isscalar(x) && (x > 0) && (x <= 1));
    addOptional(p, 'fsearch', 0, @(x) isnumeric(x) && isscalar(x));
    addOptional(p, 'verbose', 0, @(x) isnumeric(x) && isscalar(x) && ismember(x, [0, 1]));
    parse(p, varargin{:});
    f0 = p.Results.f0;
    tol = p.Results.tol;
    rate = p.Results.rate;
    fsearch = p.Results.fsearch;
    verbose = p.Results.verbose;

    % Automatic frequency estimation using FFT with parabolic interpolation
    if(f0 == 0)
        fsearch = 1;  % Auto-enable fine search when frequency is auto-estimated
        spec = abs(fft(sig));
        spec(1) = 0;  % Remove DC component
        spec = spec(1:floor(N/2));

        % Find peak bin
        [~,k0] = max(spec);

        % Parabolic interpolation: determine which neighbor is higher
        if(spec(min(max(k0+1,1),N/2)) > spec(min(max(k0-1,1),N/2)))
            r = 1;  % Right neighbor is higher
        else
            r = -1;  % Left neighbor is higher
        end

        % Refine frequency estimate using parabolic fit
        f0 = (k0-1 + r*spec(k0+r)/(spec(k0)+spec(k0+r)))/N;

    end

    % Initial 3-parameter linear least squares fit (cos, sin, dc)
    time = (0:N-1)';
    theta = 2*pi*f0*time;
    M = [cos(theta), sin(theta), ones([N,1])];
    x = linsolve(M,sig);
    A = x(1);   % Coefficient of cos(theta)
    B = x(2);   % Coefficient of sin(theta)
    dc = x(3);  % DC component

    % Iterative frequency refinement (only if fsearch is enabled)
    freq = f0;

    if(fsearch)
        delta_f = 0;

        for ii = 1:100

            % Update frequency
            freq = freq+delta_f;
            theta = 2*pi*freq*time;

            % Construct least squares matrix with frequency gradient column
            % The 4th column is the partial derivative of the signal w.r.t. frequency
            M = [cos(theta), sin(theta), ones([N,1]), (-A*2*pi*time.*sin(theta)+B*2*pi*time.*cos(theta))/N];
            x = linsolve(M,sig);
            A = x(1);
            B = x(2);
            dc = x(3);
            delta_f = x(4)*rate/N;  % Frequency update scaled by learning rate

            % Relative error in frequency update
            relerr = rms(x(4)/N*M(:,4)) / sqrt(x(1)^2+x(2)^2);

            if verbose
                fprintf('Freq iterating (%d): freq = %d, delta_f = %d, rel_err = %d\n', ii, freq, delta_f, relerr);
            end

            % Check convergence
            if(relerr < tol)
                break;
            end

        end

        % Warn if not converged
        if ii == 100 && relerr >= tol
            warning('sinfit:noConvergence', ...
                'Failed to converge in 100 iterations. Relative error = %.2e', relerr);
        end
    end

    % Generate fitted signal
    fitout = A*cos(theta)+B*sin(theta)+dc;

    % Convert to magnitude-phase form
    mag = sqrt(A^2+B^2);
    phi = -atan2(B,A);  % Negative sign for convention: mag*cos(2*pi*freq*t + phi)
end
