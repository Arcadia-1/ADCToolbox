%% Centralized Configuration for Dout Test
common_test_dout;

%% Calibration Configuration
Order = 5;  % Polynomial order for FGCalSine

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);
    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, mfilename);
    
    bits = readmatrix(dataFilePath);
    toolset_dout(bits, subFolder, 'Visible', verbose, 'Order', Order);
end