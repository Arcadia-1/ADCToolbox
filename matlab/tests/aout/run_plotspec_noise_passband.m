%% Regression checks for noise estimates over the passband (bins above the cutoff)

fprintf('[%s] noise estimates over the passband\n', mfilename);

% The true in-band noise is the power of the injected noise in the passband
% bins of its own unwindowed DFT; the spectrum under test is never used.
rngState = rng;
N = 8192;
n = 0:N-1;
toneBin = 809;
amplitude = 0.9;
nInband = N/2 + 1;
args = {'window', 'hann', 'disp', false, 'label', false};

%% White noise: no method may move when a cutoff removes a quarter of the band.
% Methods 'median' and 'mean' used to read 1.25 dB high here: they still
% scaled to every in-band bin. The tone at bin 2809 stays in the passband;
% its H3 folds to bin 235, below the cutoff. The tolerance covers the
% sampling spread of the median over 3072 rather than 4096 bins.
rng(20261110);
noise = amplitude/sqrt(2) * 1e-4 * randn(1, N);
x = amplitude * sin(2*pi*2809*n/N + 0.4) + noise;
cutoffBin = N/8;
% NSD is noise per hertz, so white noise has the same NSD whatever band it is
% integrated over; dividing the passband noise by the full band width read
% 1.25 dB low here for every method.
for method = {'auto', 'median', 'mean', 'exclude'}
    [~, ~, ~, ~, ~, ~, nfAll, nsdAll] = plotspec(x, 1, 2, 5, args{:}, 'NFMethod', method{1});
    [~, ~, ~, ~, ~, ~, nfCut, nsdCut] = plotspec(x, 1, 2, 5, args{:}, 'NFMethod', method{1}, ...
        'cutoff', (cutoffBin - 0.5)/N);
    errAll = nfAll - passband_noise_dbfs(noise, 0, nInband);
    errCut = nfCut - passband_noise_dbfs(noise, cutoffBin, nInband);
    assert(abs(errCut - errAll) < 0.15, ...
        'NFMethod=%s: noise floor error %.4f dB with the cutoff, %.4f dB without.', ...
        method{1}, errCut, errAll);
    assert(abs(nsdCut - nsdAll) < 0.15, ...
        'NFMethod=%s: NSD %.4f dB/Hz with the cutoff, %.4f dB/Hz without.', ...
        method{1}, nsdCut, nsdAll);
end

% The truth passband runs from bin max(cutoffBin, 1) to Nyquist, where the
% Nyquist bin counts half; with Fs = 1 one bin is 1/N Hz wide.
[~, ~, ~, ~, ~, ~, ~, nsd] = plotspec(x, 1, 2, 5, args{:}, 'NFMethod', 'exclude', ...
    'cutoff', (cutoffBin - 0.5)/N);
trueNsd = passband_noise_dbfs(noise, cutoffBin, nInband) - 10*log10((N/2 - cutoffBin + 0.5)/N);
assert(abs(nsd - trueNsd) < 0.1, 'NFMethod=exclude with the cutoff: NSD %.4f dB/Hz, truth %.4f dB/Hz.', ...
    nsd, trueNsd);

%% Wide harmonic lobes: 'exclude' fills them instead of dropping their noise.
% A non-coherent tone at 100 dB SNR widens sideBin 'auto' to about 120 bins,
% so the harmonic lobes cover about a quarter of the band. Without the fill
% the noise floor read 1.5 dB low; the rest of the error is noise under the
% DC and signal lobes, which SNDR leaves out as well.
rng(100);
noise = amplitude/sqrt(2) * 1e-5 * randn(1, N);
x = amplitude * sin(2*pi*(toneBin + 0.37)*n/N + 0.4) + noise;
for cutoffBin = [0, 32]
    if cutoffBin > 0
        cutoffArgs = {'cutoff', (cutoffBin - 0.5)/N};
    else
        cutoffArgs = {};
    end
    [~, ~, ~, ~, ~, ~, nf] = plotspec(x, 1, 2, 5, args{:}, 'NFMethod', 'exclude', cutoffArgs{:});
    err = nf - passband_noise_dbfs(noise, cutoffBin, nInband);
    assert(abs(err) < 0.5, 'Wide harmonic lobes, cutoff bin %d: noise floor error %.4f dB.', ...
        cutoffBin, err);
end

%% A cutoff above the band edge leaves the noise metrics undefined.
x = amplitude * sin(2*pi*101*n/N);
lastwarn('');
warnState = warning('off', 'plotspec:emptyPassband');
[enob, sndr, ~, snr, ~, ~, nf, nsd] = plotspec(x, 1, 2, 5, args{:}, 'OSR', 4, 'cutoff', 0.2);
warning(warnState);
[~, warnId] = lastwarn;
assert(strcmp(warnId, 'plotspec:emptyPassband'), 'An empty passband must raise plotspec:emptyPassband.');
assert(all(isnan([enob, sndr, snr, nf, nsd])), 'An empty passband must leave the noise metrics NaN.');
assert(isreal(nsd), 'An empty passband must not give a complex NSD.');

rng(rngState);

function nf = passband_noise_dbfs(noise, passbandStart, nInband)
% In-band power of the injected noise from bin passbandStart (0-based) up,
% in dBFS for a full scale of +/-1 (a full-scale sine has power 1/2).
    n = numel(noise);
    p = abs(fft(noise)).^2 / n^2;
    p = p(1:floor(n/2) + 1);
    p(2:end) = 2 * p(2:end);
    if mod(n, 2) == 0
        p(end) = p(end) / 2;
    end
    nf = 10*log10(2 * sum(p(max(passbandStart, 1) + 1:nInband)));
end
