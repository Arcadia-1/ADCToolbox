run(fullfile(fileparts(mfilename('fullpath')), '..', 'utils', 'ensureMatlabRoot.m'));
close all; clc; clear;

N = 2048;
Fs = 100e6;
Fin = Fs / 128;

[sig1, info1] = noiseshape('N', N, 'Fs', Fs, 'Fin', Fin, ...
    'bits', 10, 'range', [-0.5 0.5], 'order', 1);
[sig2, info2] = noiseshape(info1.clean, 'bits', 10, ...
    'range', [-0.5 0.5], 'order', 2);
[sigCustom, infoCustom] = noiseshape(info1.clean, 'bits', 10, ...
    'range', [-0.5 0.5], 'ntf', [1 -1.5 0.5]);

assert(iscolumn(sig1), 'Generated signal should be a column vector.');
assert(numel(sig1) == N, 'Generated signal length mismatch.');
assert(all(size(sig2) == size(sig1)), 'Processed signal size mismatch.');
assert(all(size(sigCustom) == size(sig1)), 'Custom NTF signal size mismatch.');
assert(isequal(info1.ntf_num, [1 -1]), '1st-order NTF coefficients mismatch.');
assert(isequal(info2.ntf_num, [1 -2 1]), '2nd-order NTF coefficients mismatch.');
assert(isequal(infoCustom.ntf_num, [1 -1.5 0.5]), 'Custom NTF coefficients mismatch.');
assert(rms(info2.shaped_error) > 0, 'Shaped error should be non-zero.');

outFolder = "test_output/test_noiseshape";
if ~isfolder(outFolder), mkdir(outFolder); end

results = table( ...
    [1; 2; 99], ...
    [rms(info1.shaped_error); rms(info2.shaped_error); rms(infoCustom.shaped_error)], ...
    'VariableNames', {'order', 'shaped_error_rms'});
writetable(results, fullfile(outFolder, 'noiseshape_results_matlab.csv'));

fprintf('noiseshape tests passed. Output saved to %s\n', outFolder);

