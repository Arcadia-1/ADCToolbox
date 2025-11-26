function saveVariable(folder, var, verbose)
if nargin < 3, verbose = 0; end
varName = inputname(2);
if isempty(varName)
    error('saveVariable must be passed a direct variable, not an expression.');
end
if ~isfolder(folder), mkdir(folder); end

% cut to 1000 rows or columns to save space
sz = size(var);
if isvector(var)
    N = min(1000, numel(var));
    var = var(1:N);
else
    [~, dim] = max(sz);
    if dim == 1
        N = min(1000, sz(1));
        var = var(1:N, :);
    else
        N = min(1000, sz(2));
        var = var(:, 1:N);
    end
end

T = table(var);
fileName = sprintf('%s_matlab.csv', varName);
filePath = fullfile(folder, fileName);
writetable(T, filePath);

if verbose
    fprintf("  [%s]->[%s]\n", mfilename, filePath);
end
end
