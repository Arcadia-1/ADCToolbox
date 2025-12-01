%% make_overview_plot.m - Create 3x3 overview plot from test results
% Combines 9 different analysis plots into a single overview figure
%
% Output: test_output/<dataset_name>/OVERVIEW_<dataset_name>_matlab.png

close all; clc; clear;

%% Configuration
inputDir = "dataset";
outputDir = "test_output";

% Test datasets - leave empty to auto-search
% You can specify a dataset name or leave empty to process all
datasetName = {};  % e.g., "sinewave_jitter_1000fs" or "" for auto-search

%% Auto-search if not specified
if isempty(datasetName)
    % Search for sinewave datasets in test_output directory
    searchResults = dir(fullfile(outputDir, 'sinewave_*'));
    searchResults = searchResults([searchResults.isdir]);  % Keep only directories

    if isempty(searchResults)
        error('No sinewave dataset directories found in %s\nPlease run the unit tests first.', outputDir);
    end

    % Use the first dataset found
    datasetName = searchResults(1).name;
    fprintf('Using dataset: %s\n', datasetName);
else
    fprintf('Using specified dataset: %s\n', datasetName);
end

%% Define plot file paths
% Based on the test output structure: test_output/<dataset_name>/test_<function>/
datasetDir = fullfile(outputDir, datasetName);

% Check if dataset directory exists
if ~isfolder(datasetDir)
    error('Dataset directory not found: %s\nPlease run the unit tests first.', datasetDir);
end

% Define the 9 plot files to include
plotFiles = {
    fullfile(datasetDir, 'test_tomDecomp', 'tomDecomp_matlab.png');           % (a) Time-domain Error Decomposition
    fullfile(datasetDir, 'test_specPlot', 'spectrum_matlab.png');             % (b) Frequency Spectrum
    fullfile(datasetDir, 'test_specPlotPhase', 'phase_matlab.png');           % (c) Phase-domain Error
    fullfile(datasetDir, 'test_errHistSine', 'errHistSine_code_matlab.png');  % (d) Error Histogram by Code
    fullfile(datasetDir, 'test_errHistSine', 'errHistSine_phase_matlab.png'); % (e) Error Histogram by Phase
    fullfile(datasetDir, 'test_errPDF', 'errPDF_matlab.png');                 % (f) Error PDF
    fullfile(datasetDir, 'test_errAutoCorrelation', 'errACF_matlab.png');     % (g) Error Autocorrelation
    fullfile(datasetDir, 'test_errSpectrum', 'errSpectrum_matlab.png');       % (h) Error Spectrum
    fullfile(datasetDir, 'test_errEnvelopeSpectrum', 'errEnvelopeSpectrum_matlab.png'); % (i) Error Envelope Spectrum
};

% Labels for each subplot
OVERVIEW_LABELS = {
    '(a) Time-domain Error Decomposition', ...
    '(b) Frequency Spectrum (dBFS)', ...
    '(c) Phase-domain Error (dB)', ...
    '(d) Error Histogram by Code (LSB)', ...
    '(e) Error Histogram by Phase (deg)', ...
    '(f) Error PDF by Magnitude (LSB)', ...
    '(g) Error Autocorrelation', ...
    '(h) Error Spectrum', ...
    '(i) Error Envelope Spectrum'
};

%% Check which files exist
fprintf('Checking for required plot files...\n');
missingFiles = {};
for i = 1:length(plotFiles)
    if isfile(plotFiles{i})
        fprintf('  [✓] %s\n', plotFiles{i});
    else
        fprintf('  [✗] Missing: %s\n', plotFiles{i});
        missingFiles{end+1} = plotFiles{i};
    end
end

if ~isempty(missingFiles)
    warning('%d plot files are missing. They will appear as blank in the overview.', length(missingFiles));
end

%% Create 3x3 overview plot
fprintf('\nCreating overview plot...\n');

fig = figure('Position', [50 50 1600 1400], 'Visible', 'off');
tlo = tiledlayout(fig, 3, 3, ...
    'TileSpacing', 'none', ...
    'Padding', 'none');

for p = 1:9
    nexttile(tlo, p);
    img_path = plotFiles{p};

    try
        if isfile(img_path)
            img = imread(img_path);
            imshow(img, 'Border', 'tight');
            axis tight;
            axis off;
            title(OVERVIEW_LABELS{p}, 'FontSize', 16, 'Interpreter', 'none');
        else
            % Display placeholder for missing files
            text(0.5, 0.5, sprintf('Missing:\n%s', img_path), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'FontSize', 10, ...
                'Color', 'red');
            axis([0 1 0 1]);
            axis off;
            title(OVERVIEW_LABELS{p}, 'FontSize', 16, 'Interpreter', 'none', 'Color', 'red');
        end
    catch ME
        % Display error message
        text(0.5, 0.5, sprintf('Error loading:\n%s\n\n%s', img_path, ME.message), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontSize', 10, ...
            'Color', 'red');
        axis([0 1 0 1]);
        axis off;
        title(OVERVIEW_LABELS{p}, 'FontSize', 16, 'Interpreter', 'none', 'Color', 'red');
    end
end

%% Save overview plot
outputFilepath = fullfile(datasetDir, sprintf('OVERVIEW_%s_matlab.png', datasetName));
exportgraphics(fig, outputFilepath, 'Resolution', 300);
fprintf('\n[Saved] %s\n', outputFilepath);
fprintf('[make_overview_plot COMPLETE]\n');

close(fig);
