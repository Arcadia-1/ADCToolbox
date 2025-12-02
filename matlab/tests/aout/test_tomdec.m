%% Centralized Configuration for Aout Test
common_test_aout;

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);
    [~, datasetName, ~] = fileparts(currentFilename);

    read_data = readmatrix(dataFilePath);
    relative_fin = findfreq(read_data);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [sine, error, harmic, others] = tomdec(read_data, relative_fin, 10, 1);
    title("Thompson decomposition");
    set(gca, "FontSize",16)

    figureName = sprintf("%s_%s_matlab.png", mfilename, datasetName);
    saveFig(figureDir, figureName, verbose);

    subFolder = fullfile(outputDir, datasetName, mfilename);  
    saveVariable(subFolder, sine, verbose);
    saveVariable(subFolder, error, verbose);
    saveVariable(subFolder, harmic, verbose);
    saveVariable(subFolder, others, verbose);
end
