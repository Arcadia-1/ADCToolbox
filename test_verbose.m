% Quick test of verbose parameter in FGCalSine
clear; clc;

% Add ADCToolbox to path
addpath('d:\ADCToolbox\matlab\src');

% Generate synthetic 10-bit ADC data
N = 8192;
M = 10;
freq_true = 0.079;
nomWeight = 2.^(M-1:-1:0);

% Create ideal binary-weighted ADC output
t = (0:N-1)';
signal = 512 * sin(2*pi*freq_true*t);
codes = round(signal + 512);
codes = max(0, min(2^M-1, codes));

% Convert to bits
bits = zeros(N, M);
for i = 1:M
    bits(:,i) = bitget(codes, M-i+1);
end

fprintf('=== Test 1: Default (verbose=0, should be silent) ===\n');
tic;
[weight1, offset1, ~, ~, ~, freq1] = FGCalSine(bits);
t1 = toc;
fprintf('Result: freq = %.6f, time = %.3f sec\n\n', freq1, t1);

fprintf('=== Test 2: With verbose=1 (should show progress) ===\n');
tic;
[weight2, offset2, ~, ~, ~, freq2] = FGCalSine(bits, 'verbose', 1);
t2 = toc;
fprintf('Result: freq = %.6f, time = %.3f sec\n\n', freq2, t2);

fprintf('=== Verification ===\n');
fprintf('Frequencies match: %s\n', isequal(freq1, freq2));
fprintf('Weights match: %s\n', isequal(weight1, weight2));
fprintf('Offsets match: %s\n\n', isequal(offset1, offset2));

fprintf('Test complete!\n');
