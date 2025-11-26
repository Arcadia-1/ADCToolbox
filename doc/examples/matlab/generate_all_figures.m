%% generate_all_figures.m
% Master script to generate all documentation figures
%
% This script:
% 1. Generates canonical datasets (if needed)
% 2. Runs all example scripts to create figures
% 3. Outputs to doc/figures/ (organized by tool)
%
% Run from: d:\ADCToolbox\doc\examples\matlab\
%
% Requirements:
% - ADCToolbox must be in MATLAB path
% - Run from doc/examples/matlab/ directory

function generate_all_figures()

close all; clc;

%% Setup paths
fprintf('=== ADCToolbox Documentation Figure Generation ===\n\n');

% Add ADCToolbox src to path
toolboxSrc = fullfile(pwd, '..', '..', '..', 'matlab', 'src');
if exist(toolboxSrc, 'dir')
    addpath(toolboxSrc);
    fprintf('[Setup] Added ADCToolbox src to path: %s\n', toolboxSrc);
else
    error('Cannot find ADCToolbox src directory. Run from doc/examples/matlab/');
end

% Add scripts to path
scriptsDir = fullfile(pwd, 'scripts');
if exist(scriptsDir, 'dir')
    addpath(scriptsDir);
else
    error('Cannot find scripts directory. Run from doc/examples/matlab/');
end

%% Step 1: Generate canonical data
fprintf('\n=== Step 1: Canonical Data Generation ===\n');
dataDir = '../data';

if ~exist(dataDir, 'dir') || isempty(dir(fullfile(dataDir, '*.csv')))
    fprintf('Generating canonical datasets...\n');
    generate_canonical_data();
else
    fprintf('Canonical datasets already exist in %s\n', dataDir);
    answer = input('Regenerate? (y/n): ', 's');
    if strcmpi(answer, 'y')
        generate_canonical_data();
    end
end

%% Step 2: Generate figures for each tool
fprintf('\n=== Step 2: Figure Generation ===\n');

% List of example functions
examples = {
    'example_FGCalSine', ...
    'example_specPlot', ...
    'example_INLsine'
};

fprintf('Will generate figures for %d tools:\n', length(examples));
for i = 1:length(examples)
    fprintf('  [%d] %s\n', i, examples{i});
end

fprintf('\nStarting generation...\n\n');

% Run each example
tic;
for i = 1:length(examples)
    fprintf('[%d/%d] Running %s...\n', i, length(examples), examples{i});
    try
        feval(examples{i});
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        fprintf('  Stack:\n');
        for j = 1:length(ME.stack)
            fprintf('    %s (line %d)\n', ME.stack(j).name, ME.stack(j).line);
        end
    end
end
elapsed = toc;

%% Summary
fprintf('\n=== Summary ===\n');
fprintf('Total time: %.2f seconds\n', elapsed);

% Count generated figures
figDir = '../../figures';
figCount = 0;
if exist(figDir, 'dir')
    allFiles = dir(fullfile(figDir, '**', '*.png'));
    figCount = length(allFiles);
end

fprintf('Generated figures: %d\n', figCount);
fprintf('Output directory: %s\n', fullfile(pwd, figDir));

fprintf('\n=== Done! ===\n');
fprintf('Figures are ready for use in documentation.\n');
fprintf('Next steps:\n');
fprintf('  1. Review figures in doc/figures/\n');
fprintf('  2. Update .md files to reference figures\n');
fprintf('  3. Commit figures to git (if desired)\n');

end
