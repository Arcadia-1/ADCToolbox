%% test_alias.m
close all; clc; clear;

%% Test Cases: [J, N, expected_bin]
testCases = [
    100,    1024,   100;
    600,    1024,   424;
    512,    1024,   512;
    1024,   1024,   0;
    1100,   1024,   76;
    2048,   1024,   0;
    2100,   1024,   52;
];

%% Run Tests
fprintf('=== test_alias.m ===\n');

nTests = size(testCases, 1);
results = zeros(nTests, 3);

for k = 1:nTests
    J = testCases(k, 1);
    N = testCases(k, 2);
    expected = testCases(k, 3);
    bin = alias(J, N);

    results(k, :) = [J, N, bin];
    pass = (bin == expected);

    fprintf('[%d/%d] alias(%4d, %4d) = %4d (expected %4d) - %s\n', ...
        k, nTests, J, N, bin, expected, char('PASS' * pass + 'FAIL' * ~pass));
end

%% Save Results
outFolder = "test_output/test_alias";
if ~isfolder(outFolder), mkdir(outFolder); end

writetable(array2table(results, 'VariableNames', {'J','N','aliased_bin'}), ...
           fullfile(outFolder, 'alias_results_matlab.csv'));
