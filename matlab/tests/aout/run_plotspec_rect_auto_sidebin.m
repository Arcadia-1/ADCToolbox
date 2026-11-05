%% Regression checks for sideBin 'auto' with a rectangular window

fprintf('[%s] rectangular-window sideBin auto checks\n', mfilename);

rngState = rng;
rng(20261105);
amplitude = 0.45;
snrInjected = 60;
args = {'window', 'rect', 'disp', false, 'label', false};

%% A coherent tone occupies one bin: sideBin 'auto' must not widen past it.
% The neighbouring bins hold only noise, so interpolating them used to move
% the tone off its bin and let the off-bin rect leakage widen sideBin to
% the search limit, which overstated SNDR by about 2 dB.
cases = [1024, 101; 8192, 809];
for ii = 1:size(cases, 1)
    N = cases(ii, 1);
    n = 0:N-1;
    sigma = amplitude / sqrt(2) * 10^(-snrInjected/20);
    x = amplitude * sin(2*pi*cases(ii, 2)*n/N + 0.4) + sigma * randn(1, N);

    [~, sndrAuto] = plotspec(x, 1, 2, 5, args{:});
    [~, sndrOneBin] = plotspec(x, 1, 2, 5, args{:}, 'sideBin', 0);

    assert(abs(sndrAuto - sndrOneBin) < 1e-9, ...
        'Coherent rect tone (N=%d): sideBin auto gives SNDR %.3f dB, sideBin 0 gives %.3f dB.', ...
        N, sndrAuto, sndrOneBin);
    assert(abs(sndrAuto - snrInjected) < 0.5, ...
        'Coherent rect tone (N=%d): SNDR %.3f dB, injected %d dB.', N, sndrAuto, snrInjected);
end

%% A noiseless ideal quantizer behaves the same way.
N = 8192;
n = 0:N-1;
lsb = 2 / 2^12;
x = (floor(0.98 * sin(2*pi*809*n/N + 0.3) / lsb) + 0.5) * lsb;
[~, sndrAuto] = plotspec(x, 1, 2, 5, args{:});
[~, sndrOneBin] = plotspec(x, 1, 2, 5, args{:}, 'sideBin', 0);
assert(abs(sndrAuto - sndrOneBin) < 1e-9, ...
    'Noiseless 12-bit quantizer: sideBin auto gives SNDR %.3f dB, sideBin 0 gives %.3f dB.', ...
    sndrAuto, sndrOneBin);

%% A non-coherent tone still widens sideBin to hold its rect leakage.
N = 1024;
n = 0:N-1;
sigma = amplitude / sqrt(2) * 10^(-snrInjected/20);
x = amplitude * sin(2*pi*101.37*n/N + 0.4) + sigma * randn(1, N);
[~, sndrAuto] = plotspec(x, 1, 2, 5, args{:});
[~, sndrOneBin] = plotspec(x, 1, 2, 5, args{:}, 'sideBin', 0);
assert(sndrAuto > sndrOneBin + 10, ...
    'Non-coherent rect tone: sideBin auto SNDR %.3f dB should exceed the one-bin %.3f dB.', ...
    sndrAuto, sndrOneBin);

rng(rngState);
