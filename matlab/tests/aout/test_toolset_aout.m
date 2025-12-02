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

    % Run toolset_aout to generate 9 individual plots
    aout_data = readmatrix(dataFilePath);
    status = toolset_aout(aout_data, subFolder, 'Visible', verbose, 'Resolution', Resolution);

    % Generate panel (3x3 overview of all 9 plots)
    if status.success
        panel_status = toolset_aout_panel(status.plot_files, subFolder, 'Visible', verbose, 'Prefix', 'aout');
    else
        fprintf('[WARNING] toolset_aout failed, skipping panel generation\n');
    end
end
