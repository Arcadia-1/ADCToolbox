%% Copy MATLAB outputs from test_output/ to test_reference/

test_output = '../../test_output';
test_reference = '../../test_reference';

% Find all MATLAB output files
datasets = dir(test_output);
datasets = datasets([datasets.isdir] & ~startsWith({datasets.name}, '.'));

fprintf('Copying MATLAB outputs to test_reference/...\n');

for d = 1:length(datasets)
    dataset = datasets(d).name;

    % Find test folders
    tests = dir(fullfile(test_output, dataset));
    tests = tests([tests.isdir] & ~startsWith({tests.name}, '.'));

    for t = 1:length(tests)
        test_name = tests(t).name;
        src_dir = fullfile(test_output, dataset, test_name);
        dst_dir = fullfile(test_reference, dataset, test_name);

        % Find MATLAB files
        files = [dir(fullfile(src_dir, '*_matlab.csv')); ...
                 dir(fullfile(src_dir, '*_matlab.png'))];

        if ~isempty(files)
            % Create destination
            if ~isfolder(dst_dir)
                mkdir(dst_dir);
            end

            % Copy files
            for f = 1:length(files)
                copyfile(fullfile(src_dir, files(f).name), ...
                        fullfile(dst_dir, files(f).name));
                fprintf('  %s/%s/%s\n', dataset, test_name, files(f).name);
            end
        end
    end
end

fprintf('Done!\n');
