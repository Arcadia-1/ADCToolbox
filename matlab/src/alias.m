function fal = alias(fin,fs)
%ALIAS Calculate aliased frequency after sampling
%   This function returns the aliased frequency of a signal after sampling,
%   accounting for different Nyquist zones. The aliasing follows the pattern
%   where signals in even Nyquist zones (0, 2, 4...) alias normally, while
%   signals in odd Nyquist zones (1, 3, 5...) alias with spectral inversion.
%
%   Syntax:
%     fal = ALIAS(fin, fs)
%
%   Inputs:
%     fin - Signal frequency before sampling
%       Scalar or Vector
%     fs - Sampling frequency. Must be a positive real number.
%       Scalar
%
%   Outputs:
%     fal - Aliased signal frequency after sampling 
%       Scalar or Vector (same size as fin)
%       Range: [0, fs/2]
%
%   Examples:
%     % Signal at 0.7*fs aliases to 0.3*fs in first Nyquist zone (mirrored)
%     fal = alias(70, 100)  % Returns 30 
%
%     % Signal at 1.3*fs aliases to 0.3*fs in second Nyquist zone (normal)
%     fal = alias(130, 100)  % Returns 30 
%
%     % Multiple frequencies
%     fal = alias([30 70 130], 100)  % Returns [30 30 30]
%
%   Notes:
%     - Nyquist zone n is defined as [(n)*fs/2, (n+1)*fs/2]
%     - Even zones: direct aliasing (fal = mod(fin,fs))
%     - Odd zones: mirrored aliasing (fal = fs - mod(fin,fs))
%
%   See also: findFin, findBin

    % Input validation
    if fs <= 0
        error('alias:invalidFs', 'Sampling frequency fs must be positive.');
    end

    if(~isreal(fin) || ~isreal(fs))
        error('alias:invalidInput', 'Frequencies must be real numbers.');
    end

    % Determine Nyquist zone (0-based indexing)
    % Zone 0: [0, fs/2], Zone 1: [fs/2, fs], Zone 2: [fs, 3*fs/2], etc.
    nyquistZone = floor(fin / fs * 2);

    % Calculate base frequency offset within the Nyquist zone
    baseOffset = fin - floor(fin / fs) * fs;

    % Apply aliasing rule based on even/odd Nyquist zone
    % Even zones (0,2,4,...): normal aliasing
    % Odd zones (1,3,5,...): mirrored aliasing
    isEvenZone = mod(nyquistZone, 2) == 0;
    fal = isEvenZone .* baseOffset + ~isEvenZone .* (fs - baseOffset);

end

