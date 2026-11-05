%% Regression checks for the default SNR noise estimate (NFMethod='exclude')

fprintf('[%s] default SNR against an injected ground truth\n', mfilename);

% The truth is a time-domain least-squares fit of DC and harmonics 1..5 at the
% known frequency: the fundamental gives the signal power and the residual,
% which holds everything except DC and H1..H5, is the noise.
N = 8192;
n = 0:N-1;
k = 809;
rngState = rng;
rng(20261105);

%% Noiseless ideal quantizer: quantization error is the noise.
for bits = [8, 12, 16]
    lsb = 2 / 2^bits;
    x = (floor(0.98 * sin(2*pi*k*n/N + 0.3) / lsb) + 0.5) * lsb;
    for win = {'hann', 'rect'}
        [~, sndr, ~, snr] = plotspec(x, 1, 2, 5, 'window', win{1}, 'disp', false, 'label', false);
        truth = plotspec_truth_snr(x, k/N);
        assert(abs(snr - truth) < 0.1, ...
            'Default SNR of a noiseless %d-bit quantizer (%s): %.3f dB, truth %.3f dB.', bits, win{1}, snr, truth);
        assert(snr >= sndr, 'Default SNR must not fall below SNDR.');
    end
end

%% Noiseless tone with a few spurs: the spurs are the noise.
x = 0.9 * sin(2*pi*k*n/N + 0.4) ...
    + 0.9 * 10^(-40/20) * sin(2*pi*(N/4 - k)*n/N + 0.2) ...
    + 0.9 * 10^(-45/20) * sin(2*pi*(N/4 + k)*n/N + 0.9) ...
    + 0.9 * 10^(-50/20) * sin(2*pi*(N/2 - k)*n/N + 1.7);
for win = {'hann', 'rect'}
    [~, sndr, ~, snr] = plotspec(x, 1, 2, 5, 'window', win{1}, 'disp', false, 'label', false);
    truth = plotspec_truth_snr(x, k/N);
    assert(abs(snr - truth) < 0.1, ...
        'Default SNR with noiseless spurs (%s): %.3f dB, truth %.3f dB.', win{1}, snr, truth);
    assert(snr >= sndr, 'Default SNR must not fall below SNDR.');
end

%% White noise with non-harmonic spurs: spurs count as noise.
x = 0.9 * sin(2*pi*k*n/N + 0.4) ...
    + 0.9 * 10^(-75/20) * sin(2*pi*1777*n/N + 0.3) ...
    + 0.9 * 10^(-85/20) * sin(2*pi*3001*n/N + 1.1) ...
    + 0.9/sqrt(2) * 10^(-80/20) * randn(1, N);
[~, sndr, ~, snr] = plotspec(x, 1, 2, 5, 'window', 'hann', 'disp', false, 'label', false);
truth = plotspec_truth_snr(x, k/N);
assert(abs(snr - truth) < 0.2, 'Default SNR with spurs on white noise: %.3f dB, truth %.3f dB.', snr, truth);
assert(snr >= sndr, 'Default SNR must not fall below SNDR.');

rng(rngState);

function snr = plotspec_truth_snr(x, freq)
% SNR from a time-domain fit of DC and harmonics 1..5 at a known frequency.
    t = (0:numel(x)-1).';
    basis = ones(numel(x), 1);
    for h = 1:5
        basis = [basis, sin(2*pi*h*freq*t), cos(2*pi*h*freq*t)]; %#ok<AGROW>
    end
    coef = basis \ x(:);
    resid = x(:) - basis * coef;
    snr = 10*log10((coef(2)^2 + coef(3)^2) / 2 / mean(resid.^2));
end
