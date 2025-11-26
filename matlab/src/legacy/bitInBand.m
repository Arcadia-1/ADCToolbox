function [signalOut] = bitInBand(signalIn, bands)
%BITINBAND Filter signal to retain only specified frequency bands
%   This function applies FFT-based brickwall filters to extract "in-band"
%   frequency components from input signals. Each column of the input is
%   filtered independently using the same band specifications.
%
%   Syntax:
%     signalOut = BITINBAND(signalIn, bands)
%
%   Inputs:
%     signalIn - Input signal matrix where each column is filtered independently
%       Matrix (N x M) or Vector
%       If input is a row vector, it will be transposed to column vector
%     bands - Frequency band specifications (normalized to sampling frequency)
%       Matrix (P x 2) where each row [fLow, fHigh] defines a passband
%       Frequencies are normalized: 0 = DC, 0.5 = Nyquist frequency
%       Example: [0.1, 0.2; 0.3, 0.4] passes two bands
%
%   Outputs:
%     signalOut - Filtered signal with only in-band components retained
%       Matrix (N x M) - same size as signalIn (after any transpose)
%       Signal is real-valued (imaginary components from FFT are discarded)
%
%   Examples:
%     % Filter to retain only 0.1 to 0.2 normalized frequency band
%     filtered = bitInBand(signal, [0.1, 0.2])
%
%     % Filter with multiple bands
%     filtered = bitInBand(signal, [0.05, 0.15; 0.25, 0.35])
%
%     % Multi-column input (each column filtered independently)
%     filtered = bitInBand([signal1, signal2], [0.1, 0.3])
%
%   Notes:
%     - Uses FFT-based brickwall filter (sharp transitions, may cause ringing)
%     - Maintains signal symmetry for real-valued signals
%     - DC (0 Hz) and Nyquist (fs/2) components handled specially
%     - Row vectors are automatically transposed to column vectors
%
%   See also: alias, fft, ifft

    % Get input dimensions and ensure column-wise orientation
    [N, M] = size(signalIn);
    if N < M
        signalIn = signalIn';
        [N, M] = size(signalIn);
    end

    % Validate bands parameter
    [numBands, numCols] = size(bands);
    if numCols ~= 2
        error('bitInBand:invalidBands', 'bands must have exactly 2 columns [fLow, fHigh]');
    end

    if any(bands(:) < 0) || any(bands(:) > 0.5)
        error('bitInBand:invalidFrequency', 'Band frequencies must be in range [0, 0.5]');
    end

    % Compute FFT of all columns
    spectrum = fft(signalIn);

    % Initialize frequency mask (0 = reject, 1 = pass)
    mask = zeros(N, 1);

    % Build mask for each specified frequency band
    for bandIdx = 1:numBands
        % Convert normalized frequencies to bin indices
        binStart = round(min(bands(bandIdx, :)) * N);
        binEnd = round(max(bands(bandIdx, :)) * N);

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
    signalOut = real(ifft(spectrum));

end