%ENSUREMATLABROOT Change current folder to the matlab/ package root.
%   Test and data-generation scripts use relative paths such as
%   test_dataset/ and test_output/. Calling them via run('subdir/script.m')
%   changes pwd to that subfolder; invoke this helper first so paths resolve.

p = fileparts(mfilename('fullpath'));
for k = 1:8
    if isfolder(fullfile(p, 'src')) && ...
            (isfolder(fullfile(p, 'tests')) || isfolder(fullfile(p, 'data_generation')))
        matlabRoot = p;
        addpath(genpath(fullfile(matlabRoot, 'src')));
        addpath(genpath(fullfile(matlabRoot, 'tests')));
        if isfolder(fullfile(matlabRoot, 'data_generation'))
            addpath(fullfile(matlabRoot, 'data_generation'));
        end
        if ~strcmp(pwd, matlabRoot)
            cd(matlabRoot);
        end
        return;
    end
    pNext = fileparts(p);
    if isequal(pNext, p)
        break;
    end
    p = pNext;
end

error('ensureMatlabRoot:notFound', ...
    'Could not locate matlab/ root (expected src/ and tests/ or data_generation/).');
