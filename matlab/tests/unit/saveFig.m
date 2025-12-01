function saveFig(folder, pngFilename, verbose)
% saveFig Save the current figure to a PNG file and optionally print a message.

if ~isfolder(folder), mkdir(folder); end

filePath = fullfile(folder, pngFilename);
saveas(gcf, filePath);

if verbose
    fprintf("  [%s]->[%s]\n", mfilename, filePath);

    pause(1)
end
close all;
end
