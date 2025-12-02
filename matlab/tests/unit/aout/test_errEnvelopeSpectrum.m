%% test_errEnvelopeSpectrum.m - Unit test for errEnvelopeSpectrum function
close all; clc; clear;

%% Configuration
verbose = 0;
inputDir = "dataset/aout/sinewave";
outputDir = "test_output";

filesList ={};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end
%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    [~, datasetName, ~] = fileparts(currentFilename);
    titleString = replace(datasetName, '_', '\_');

    [data_fit, ~, ~, ~, ~] = sineFit(read_data);
    err_data = read_data - data_fit;

    e = err_data(:);
    env = abs(hilbert(e));

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    errEnvelopeSpectrum(err_data, 'Fs', 1);
    title(['errEnvelopeSpectrum: ', titleString]);

    subFolder = fullfile(outputDir, datasetName, mfilename);

    saveFig(subFolder, "errEnvelopeSpectrum_matlab.png", verbose);

    N_env = length(env);
    env_fft = fft(env);
    env_fft = env_fft(1:floor(N_env/2)+1);
    freq_bins = (0:length(env_fft)-1)';
    env_mag = abs(env_fft);
    env_mag_dB = 20*log10(env_mag + eps);

    envelopeTable = table(freq_bins, env_mag, env_mag_dB, ...
        'VariableNames', {'bin', 'magnitude', 'magnitude_dB'});
    envelopePath = fullfile(subFolder, 'envelope_spectrum_data_matlab.csv');
    writetable(envelopeTable, envelopePath);
end
