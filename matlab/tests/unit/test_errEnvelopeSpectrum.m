%% test_errEnvelopeSpectrum.m - Unit test for errEnvelopeSpectrum function
% Tests the errEnvelopeSpectrum function with sinewave error data
%
% Output structure:
%   test_output/<data_set_name>/test_errEnvelopeSpectrum/
%       errEnvelopeSpectrum_matlab.png
%       envelope_spectrum_data_matlab.csv   - Envelope spectrum data

close all; clc; clear;

%% Configuration
inputDir = "dataset";
outputDir = "test_output";

% Test datasets - leave empty to auto-search
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% Test Loop
fprintf('=== test_errEnvelopeSpectrum.m ===\n');
fprintf('[Testing] %d datasets...\n\n', length(filesList));

for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);

    if ~isfile(dataFilePath)
        fprintf('[%d/%d] %s - NOT FOUND, skipping\n\n', k, length(filesList), currentFilename);
        continue;
    end
    fprintf('[%d/%d] [Processing] %s\n', k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    % Extract dataset name
    [~, datasetName, ~] = fileparts(currentFilename);
    titleString = replace(datasetName, '_', '\_');

    % Create output subfolder
    subFolder = fullfile(outputDir, datasetName, 'test_errEnvelopeSpectrum');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    %% Compute error data using sineFit
    [data_fit, ~, ~, ~, ~] = sineFit(read_data);
    err_data = read_data - data_fit;

    %% Compute envelope spectrum data
    e = err_data(:);
    env = abs(hilbert(e)); % envelope = |hilbert|

    %% Run errEnvelopeSpectrum (for plotting)
    figure('Visible', 'off');
    errEnvelopeSpectrum(err_data, 'Fs', 1);
    title(['errEnvelopeSpectrum: ', titleString]);

    % Save plot
    plotPath = fullfile(subFolder, 'errEnvelopeSpectrum_matlab.png');
    saveas(gcf, plotPath);
    fprintf('  [Saved] %s\n', plotPath);
    close(gcf);

    % Save envelope spectrum data to CSV
    % Get the spectrum of the envelope
    N_env = length(env);
    env_fft = fft(env);
    env_fft = env_fft(1:floor(N_env/2)+1); % One-sided spectrum
    freq_bins = (0:length(env_fft)-1)';
    env_mag = abs(env_fft);
    env_mag_dB = 20*log10(env_mag + eps); % Add eps to avoid log(0)

    envelopeTable = table(freq_bins, env_mag, env_mag_dB, ...
        'VariableNames', {'bin', 'magnitude', 'magnitude_dB'});
    envelopePath = fullfile(subFolder, 'envelope_spectrum_data_matlab.csv');
    writetable(envelopeTable, envelopePath);
    fprintf('  [Saved] %s\n\n', envelopePath);
end

fprintf('[test_errEnvelopeSpectrum COMPLETE]\n');
