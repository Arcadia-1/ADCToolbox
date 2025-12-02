function snr = ntfperf(ntf,fl,fh,disp)
%NTFPERF Analyze noise transfer function performance
%   This function evaluates the noise suppression performance of a noise
%   transfer function (NTF) within a specified signal band. It calculates
%   the SNR improvement in the specified band compared to the baseline
%   (non-shaping and non-oversampling) in dB.
%
%   Syntax:
%     snr = NTFPERF(ntf, fl, fh)
%     snr = NTFPERF(ntf, fl, fh, disp)
%
%   Inputs:
%     ntf - Noise transfer function in z-domain
%       Transfer function object (tf, zpk, or ss)
%     fl - Low bound frequency of signal band (relative to Fs)
%       Scalar in range [0, 0.5]
%     fh - High bound frequency of signal band (relative to Fs)
%       Scalar in range (fl, 0.5] (fh must be greater than fl)
%     disp - Display plot switch (optional, default: 0)
%       Logical or numeric (0 or 1)
%       When set to 1, plots NTF magnitude response with signal band markers
%
%   Outputs:
%     snr - SNR improvement from noise-shaping and over-sampling (dB)
%       Scalar
%       Positive values indicate SNR improvement relative to the baseline case
%
%   Examples:
%     % Analyze an ideal 1st-order lowpass delta-sigma modulator's NTF
%     ntf = tf([1 -1], [1 0], 1);     % 1st-order NTF (1-z^-1)
%     snr = ntfperf(ntf, 0, 0.5/16)   % SNR improvement under 16x OSR (~31dB)
%
%     % Analyze bandpass NTF with visualization
%     ntf = tf([1 0 1], [1 0 0], 1);        % bandpass NTF (1+z^-2)
%     snr = ntfperf(ntf, 0.24, 0.26, 1)     % Shows plot of NTF response
%
%   Notes:
%     - The function evaluates NTF at 1E6 frequency points for accuracy
%     - SNR improvement is computed using integrated squared magnitude
%     - Plot automatically displays when disp=1 or when no output arguments
%     - For lowpass (fl=0): uses semilogx scale, shows only fh marker
%     - For bandpass (fl>0): uses linear scale, shows both fl and fh markers
%
%   See also: bode, zpk, tf, ss

    % Input validation
    if nargin < 3
        error('ntfperf:notEnoughInputs', ...
            'At least 3 input arguments required: ntf, fl, fh');
    end

    if fl < 0 || fl >= 0.5
        error('ntfperf:invalidFl', ...
            'Low frequency fl must be in range [0, 0.5)');
    end

    if fh <= fl || fh > 0.5
        error('ntfperf:invalidFh', ...
            'High frequency fh must be in range (fl, 0.5]');
    end

    % Generate frequency vector (normalized to [0, 0.5])
    N = 1e6;
    w = (1:N)/N/2;

    % Evaluate NTF magnitude response
    [mag, ~] = bode(ntf, w*2*pi);
    mag = reshape(mag, size(w));

    % Calculate integrated noise power in signal band
    bandMask = w > fl & w < fh;
    np = sum(mag(bandMask).^2) / length(w);

    % Compute SNR improvement relative to baseline (NTF=1)
    snr = -10*log10(np);

    % Plot NTF response if disp=1 or no output arguments requested
    if (nargin == 4 && disp == 1) || (nargin < 4 && nargout == 0)

        % Plot NTF magnitude response with signal band markers
        yLimits = 20*log10([min(mag)/10, max(mag)*10]);
        if fl > 0   % Bandpass case
            plot(w, 20*log10(mag));
            hold on;
            plot([fl, fl], yLimits, 'k--');
            plot([fh, fh], yLimits, 'k--');
        else
            semilogx(w, 20*log10(mag));
            hold on;
            semilogx([fh, fh], yLimits, 'k--');
        end

        ylim(yLimits);

        xlabel('Normalized Frequency');
        ylabel('Magnitude (dB)');
        title(sprintf('Noise Transfer Function (SNR += %.2fdB)',snr));
        grid on;
        hold off;
    end

end