%% test_alias.m - Unit test for alias function
% Output: test_output/test_alias/alias_results_matlab.csv

close all; clc; clear;

%% Configuration
outputDir = "test_output";

%% Test Cases - J (input freq), N (sampling freq), expected bin
testCases = [
    100,    1024,   100;    % No aliasing (J < N/2)
    600,    1024,   424;    % Aliasing (J > N/2)
    512,    1024,   512;    % Nyquist
    1024,   1024,   0;      % At sample rate
    1100,   1024,   76;     % Above sample rate
    2048,   1024,   0;      % 2x sample rate
    2100,   1024,   52;     % Above 2x sample rate
];

%% Run Tests
fprintf('=== test_alias.m ===\n');
fprintf('[Testing] %d cases...\n\n', size(testCases, 1));

results = zeros(size(testCases, 1), 4);
allPass = true;

for k = 1:size(testCases, 1)
    J = testCases(k, 1);
    N = testCases(k, 2);
    expected = testCases(k, 3);
    bin = alias(J, N);

    results(k, :) = [J, N, bin, expected];

    if bin == expected
        status = 'PASS';
    else
        status = 'FAIL';
        allPass = false;
    end

    fprintf('[%d/%d] alias(%4.1d, %4.1d) = %4.1d (expected %4.1d) - %s\n', ...
        k, size(testCases, 1), J, N, bin, expected, status);
end

%% Save Results
subFolder = fullfile(outputDir, 'test_alias');
if ~isfolder(subFolder), mkdir(subFolder); end

csvPath = fullfile(subFolder, 'alias_results_matlab.csv');
writetable(array2table(results, 'VariableNames', {'J','N','bin','expected'}), csvPath);

fprintf('\n  [Saved] %s\n', csvPath);
if allPass
    fprintf('\n[test_alias PASSED]\n');
else
    fprintf('\n[test_alias FAILED - Some tests failed]\n');
end
