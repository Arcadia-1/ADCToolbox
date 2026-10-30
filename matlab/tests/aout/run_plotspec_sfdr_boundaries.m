%% Regression checks for SFDR lobes clipped by analysis masks

fprintf('[%s] SFDR boundary and auto-sideBin checks\n', mfilename);

N = 1024;
n = 0:N-1;

%% A noise-free coherent Hann tone must retain the Hann main lobe.
pureTone = sin(2*pi*301*n/N);
[~, sndrPure, sfdrPure] = plotspec( ...
    pureTone, 100e6, 2, 5, ...
    'window', 'hann', ...
    'sideBin', 'auto', ...
    'disp', false, ...
    'label', false);

assert(sndrPure > 200, 'Auto sideBin collapsed and left Hann signal power in SNDR noise.');
assert(sfdrPure > 200, 'Auto sideBin collapsed and treated a Hann side bin as a spur.');

%% Explicit sideBin=1 must compensate spur lobes truncated by masks/boundaries.
cases = [ ...
    509, 507; ... % lower fundamental-mask edge
    301, 303; ... % upper fundamental-mask edge
    301,   1; ... % DC edge
    301, 511; ... % Nyquist edge
    509, 511; ... % both side bins unavailable
    301, 305  ... % full-lobe control
];

for ii = 1:size(cases, 1)
    fundamentalBin = cases(ii, 1);
    spurBin = cases(ii, 2);
    signal = sin(2*pi*fundamentalBin*n/N) ...
        + 1e-4 * sin(2*pi*spurBin*n/N);

    [~, ~, sfdr] = plotspec( ...
        signal, 100e6, 2, 5, ...
        'window', 'hann', ...
        'sideBin', 1, ...
        'disp', false, ...
        'label', false);

    assert(abs(sfdr - 80) < 0.01, ...
        'SFDR boundary compensation failed for bins %d and %d: %.6f dBc', ...
        fundamentalBin, spurBin, sfdr);
end

