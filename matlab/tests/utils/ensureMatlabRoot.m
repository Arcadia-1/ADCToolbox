function matlabRoot = ensureMatlabRoot()
%ENSUREMATLABROOT Change current folder to the matlab/ package root.
%   Test and data-generation scripts use relative paths such as
%   test_dataset/ and test_output/. Calling them via run('subdir/script.m')
%   changes pwd to that subfolder; invoke this helper first so paths resolve.
%
%   Also callable as a script: run(.../ensureMatlabRoot.m)

p = fileparts(mfilename('fullpath'));
for k = 1:8
    if isfolder(fullfile(p, 'src')) && ...
            (isfolder(fullfile(p, 'tests')) || isfolder(fullfile(p, 'data_generation')))
        matlabRoot = p;
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

end
