%% Regression checks for wcalsin rank-deficiency patch edge cases

fprintf('[%s] rank patch edge cases\n', mfilename);

N = 64;
M = 5;
nomWeight = 2.^(M-1:-1:0);

%% All-constant captures should fail clearly instead of reaching the solver.
bits_const = ones(N, M);
try
    wcalsin(bits_const, 'freq', 1/N, 'nomWeight', nomWeight, 'verbose', 0);
    error('run_wcalsin_rank_patch:ExpectedFailure', ...
        'Expected all-constant bits to fail rank patching.');
catch ME
    assert(contains(ME.message, 'Patched bits are empty'), ...
        'All-constant bits should report that no valid columns remain.');
end

%% Partially constant captures should keep active columns and zero unobservable bits.
n = (0:N-1)';
freq = 7/N;
active_bit = double(sin(2*pi*freq*n) > 0);
bits_partial = [
    active_bit, ...
    ones(N, 1), ...
    zeros(N, 1), ...
    ones(N, 1), ...
    zeros(N, 1)
];

[weight, ~, ~, ~, ~, ~] = wcalsin(bits_partial, ...
    'freq', freq, ...
    'nomWeight', nomWeight, ...
    'verbose', 0);

assert(abs(weight(1)) > 1e-6, 'The active bit should retain a fitted weight.');
assert(all(abs(weight(2:end)) < 1e-12), ...
    'Constant bit columns should map to zero fitted weights.');
