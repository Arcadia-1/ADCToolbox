function filesList = autoSearchFiles(filesList, inputDir, varargin)
% autoSearchFiles - Auto-search for files if filesList is empty
%
% Syntax:
%   filesList = autoSearchFiles(filesList, inputDir, pattern1, pattern2, ...)
%
% Inputs:
%   filesList - Cell array of filenames (if empty, auto-search is performed)
%   inputDir  - Directory to search in
%   varargin  - Variable number of search patterns (e.g., 'sinewave_*.csv')
%
% Outputs:
%   filesList - Cell array of discovered filenames
%
% Example:
%   filesList = {};
%   filesList = autoSearchFiles(filesList, 'test_data', 'sinewave_*.csv');

if ~isempty(filesList)
    return; % User manually specified files, skip auto-search
end

searchPatterns = varargin;
filesList = {};

for i = 1:length(searchPatterns)
    searchResults = dir(fullfile(inputDir, searchPatterns{i}));
    filesList = [filesList; {searchResults.name}'];
end

fprintf('[Auto-discovered] %d files matching patterns: %s\n', ...
    length(filesList), strjoin(searchPatterns, ', '));

if isempty(filesList)
    error('No test files found in %s', inputDir);
end
end
