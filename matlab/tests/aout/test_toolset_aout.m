%% Centralized Configuration for Aout Test
common_test_aout;

%% ADC Configuration
Resolution = 11;  % ADC resolution in bits

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);
    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(figureDir, datasetName);

    aout_data = readmatrix(dataFilePath);
    plot_files = toolset_aout(aout_data, subFolder, 'Visible', verbose, 'Resolution', Resolution);
    panel_status = toolset_aout_panel(subFolder, 'Visible', verbose, 'Prefix', 'aout');

end
