function saveFig(folder, pngFilename, verbose)
if ~isfolder(folder), mkdir(folder); end

filePath = fullfile(folder, pngFilename);
saveas(gcf, filePath);
pause(1)
if verbose
    fprintf("  [%s]->[%s]\n", mfilename, filePath);
end

% close all;
end
