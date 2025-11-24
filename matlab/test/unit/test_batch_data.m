%% ADC Diagnostic Batch Processing Script (Optimized File Structure)
% Goal: Read CSV data, process it, generate two separate plots, 
%       and save both into a DEDICATED SUBFOLDER per input file.

close all; clc; clear; 

% --- 1. Configuration ---
inputDir = "ADCToolbox_example_data";
outputDir = "ADCToolbox_example_output";

% Define the list of files to process
filesList = {
    'batch_sinewave_Nrun_2.csv';
    'batch_sinewave_Nrun_16.csv'; 
    'batch_sinewave_Nrun_100.csv'; 
};

% Check if the main output directory exists, create it if not
if ~isfolder(outputDir)
    mkdir(outputDir);
end

% --- 2. Processing Loop ---
fprintf('Starting batch processing (%d files)...\n', length(filesList));
for k = 1:length(filesList)
    currentFilename = filesList{k};
    fprintf('Processing file: %s\n', currentFilename);
    
    % --- Data Loading & Processing ---
    dataFilePath = fullfile(inputDir, currentFilename);
    read_data = readmatrix(dataFilePath); 
    
    % Extract file name components for titling and creating subfolder
    [~, name, ~] = fileparts(currentFilename);
    titleString = replace(name, '_', '\_'); % Escape underscore for plot title
    
    % --- CREATE DEDICATED SUBFOLDER ---
    subFolder = fullfile(outputDir, name); % Subfolder name is the file name (without extension)
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end
    
    
    % ---------------------------------------------------------------------
    % --- PLOT 1: SPECTRUM ---
    % ---------------------------------------------------------------------
    figure; % Open the first figure
    
    % Standard Spectrum Plot
    specPlot(read_data, 'label', 1, 'harmonic', 0, 'OSR', 1, 'coAvg', 0);
    title(['Spectrum Plot: ', titleString]);
    
    % Saving Plot 1 to subfolder
    % Using 'Spectrum_of_' prefix as you suggested for clarity
    outputFileName1 = ['Spectrum_of_', name, '_matlab.png'];
    outputFilePath1 = fullfile(subFolder, outputFileName1); 
    saveas(gcf, outputFilePath1); 
    fprintf('[Saved image 1] -> [%s]\n', outputFilePath1);
    close(gcf); % Close the figure 1
    
    
    % ---------------------------------------------------------------------
    % --- PLOT 2: PHASE ---
    % ---------------------------------------------------------------------
    figure; % Open the second, separate figure
    
    % Phase Plot
    specPlotPhase(read_data, 'harmonic', 50);
    title(['Phase Plot: ', titleString]);
    
    % Saving Plot 2 to subfolder
    % Using 'Phase_of_' prefix for clarity
    outputFileName2 = ['Phase_of_', name, '_matlab.png'];
    outputFilePath2 = fullfile(subFolder, outputFileName2); 
    saveas(gcf, outputFilePath2); 
    fprintf('[Saved image 2] -> [%s]\n\n', outputFilePath2);
    close(gcf); % Close the figure 2
end
fprintf('Batch processing complete.\n');