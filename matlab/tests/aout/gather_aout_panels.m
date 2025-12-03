%% Gather existing AOUT plots into panel figures (without regenerating plots)
common_test_aout;

%% Loop through datasets and gather panels
for k = 1:length(filesList)
    currentFilename = filesList{k};
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);
    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(figureDir, datasetName);

    % Check if folder exists
    if ~isfolder(subFolder)
        fprintf('  ✗ Skipping - folder not found: %s\n', subFolder);
        continue;
    end

    % Generate panel (auto-detects plot files)
    panel_status = toolset_aout_panel(subFolder, 'Visible', verbose, 'Prefix', 'aout');
end