function [sigout] = ifilter(sigin, passband)
%IFILTER Filter signal to retain only specified frequency bands
%   This function applies ideal filters (FFT-based brickwall filters) to extract the
%   frequency components in passband from input signals. Each column of the sigin matrix 
%   is filtered independently by the same passband.
%
%   Syntax:
%     sigout = IFILTER(sigin, passband)
%
%   Inputs:
%     sigin - Input signal matrix where each column is filtered independently
%       Matrix (N x M) or Vector
%       Must be real-valued (complex signals are not supported)
%       If input is a row vector, it will be transposed to column vector automatically
%     passband - Frequency band specifications (normalized to sampling frequency)
%       Matrix (P x 2) where each row [fLow, fHigh] defines a passband
%       The finalized passband is the union of all P passbands,
%           i.e., [fLow1, fHigh1] U [fLow2, fHigh2] U ... U [fLowP, fHighP].
%       Frequencies are normalized: 0 = DC, 0.5 = Nyquist frequency
%       Example: [0.1, 0.2; 0.3, 0.4] defines two passbands: [0.1, 0.2] and [0.3, 0.4]
%
%   Outputs:
%     sigout - Filtered signals with only in-band components retained
%       Matrix (N x M) - same size as sigin (after any transpose)
%       The filtered signals are also real-valued
%
%   Examples:
%     % Filter the sigin to retain only from 0.1 Fs to 0.2 Fs
%     sigout = ifilter(sigin, [0.1, 0.2])
%
%     % Filter with two passbands: [0.05, 0.15] and [0.25, 0.35] of Fs
%     sigout = ifilter(sigin, [0.05, 0.15; 0.25, 0.35])
%
%     % Multi-signal input case. All signals are filtered by the passband of [0.1, 0.2] Fs
%     [sigout1, sigout2, ...] = ifilter([sigin1, sigin2, ...], [0.1, 0.2])
%
%   Notes:
%     - Uses FFT-based brickwall filter (sharp transitions, may cause Gibbs effect)
%     - Transposes sigin automatically if it is a "wide" matrix (rows < cols)
%
%   See also: alias, fft, ifft

    % Validate that input signal is real
    if ~isreal(sigin)
        error('ifilter:complexInput', 'Input signal must be real-valued (not complex)');
    end

    % Get input dimensions and ensure column-wise orientation
    [N, M] = size(sigin);
    if N < M
        sigin = sigin';
        [N, M] = size(sigin);
    end

    % Validate passband parameter
    [numBands, numCols] = size(passband);
    if numCols ~= 2
        error('ifilter:invalidPassband', 'Passband must have exactly 2 columns [fLow, fHigh]');
    end

    if ~isreal(passband) || any(passband(:) < 0) || any(passband(:) > 0.5)
        error('ifilter:invalidFrequency', 'Band frequencies must be real and in range [0, 0.5]');
    end

    % Compute FFT of all columns
    spectrum = fft(sigin);

    % Initialize frequency mask (0 = reject, 1 = pass)
    mask = zeros(N, 1);

    % Build mask for each specified frequency band
    for bandIdx = 1:numBands
        % Convert normalized frequencies to bin indices
        binStart = round(min(passband(bandIdx, :)) * N);
        binEnd = round(max(passband(bandIdx, :)) * N);

        % Get aliased bin indices for this frequency range
        binIndices = alias(binStart:binEnd, N);

        % Set positive frequency components (1-based indexing)
        mask(binIndices + 1) = 1;

        % Set corresponding negative frequency components for symmetry
        % Exclude DC (binIndices=0) and Nyquist (binIndices=N/2 for even N)
        % to avoid out-of-bounds and maintain proper Hermitian symmetry
        validIndices = binIndices(binIndices > 0 & binIndices < N/2);
        mask(N - validIndices + 1) = 1;
    end

    % Apply mask to all columns of spectrum
    spectrum = spectrum .* (mask * ones(1, M));

    % Convert back to time domain
    % Use real() to discard negligible imaginary parts from numerical errors
    sigout = real(ifft(spectrum));

end
