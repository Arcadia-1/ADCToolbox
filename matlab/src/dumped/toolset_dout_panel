function status = toolset_dout_panel(outputDir, varargin)
%TOOLSET_DOUT_PANEL Gather 6 DOUT toolset plots into a 3x2 panel figure
%
%   status = toolset_dout_panel(outputDir)
%   status = toolset_dout_panel(outputDir, 'Prefix', 'dout')
%   status = toolset_dout_panel(outputDir, 'PlotFiles', plotFiles)
%
% Inputs:
%   outputDir  - Directory containing the plot files
%
% Optional Parameters:
%   'Visible'   - Show figure (default: false)
%   'Prefix'    - Filename prefix (default: 'dout')
%   'PlotFiles' - Cell array of 6 PNG file paths (auto-detected if not provided)
%
% Outputs:
%   status - Struct with fields:
%            .success (true if panel created successfully)
%            .panel_path (path to panel figure)
%            .errors (cell array of error messages)
%
% Example:
%   % Auto-detect plot files
%   panel_status = toolset_dout_panel('output/test1', 'Prefix', 'dout');
%
%   % Or provide explicit file paths
%   panel_status = toolset_dout_panel('output/test1', 'PlotFiles', plot_files);

p = inputParser;
addParameter(p, 'Visible', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'Prefix', 'dout', @ischar);
addParameter(p, 'PlotFiles', {}, @iscell);
parse(p, varargin{:});

status = struct('success', false, 'panel_path', '', 'errors', {{}});

% Auto-construct plot file paths if not provided
if isempty(p.Results.PlotFiles)
    plotFiles = {
        fullfile(outputDir, sprintf('%s_1_spectrum_nominal.png', p.Results.Prefix));
        fullfile(outputDir, sprintf('%s_2_spectrum_calibrated.png', p.Results.Prefix));
        fullfile(outputDir, sprintf('%s_3_bitActivity.png', p.Results.Prefix));
        fullfile(outputDir, sprintf('%s_4_overflowChk.png', p.Results.Prefix));
        fullfile(outputDir, sprintf('%s_5_weightScaling.png', p.Results.Prefix));
        fullfile(outputDir, sprintf('%s_6_ENoB_sweep.png', p.Results.Prefix));
    };
else
    plotFiles = p.Results.PlotFiles;
end

if length(plotFiles) ~= 6
    error('[Panel generation failed: Expected 6 plot files, got %d]', length(plotFiles));
end

fprintf('[Panel]');
try
    plotLabels = {'(1) Spectrum (Nominal)'; '(2) Spectrum (Calibrated)'; ...
        '(3) Bit Activity'; '(4) Overflow Check'; ...
        '(5) Weight Scaling'; '(6) ENoB Sweep'};

    visOpts = {'off', 'on'};
    fig = figure('Position', [10 10 1200 1000], 'Visible', visOpts{p.Results.Visible+1});
    tlo = tiledlayout(fig, 3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    for i = 1:6
        nexttile(tlo, i);
        if isfile(plotFiles{i})
            imshow(imread(plotFiles{i}), 'Border', 'tight');
            axis tight off;
            title(plotLabels{i}, 'FontSize', 12, 'Interpreter', 'none');
        else
            text(0.5, 0.5, sprintf('Missing:\n%s', plotLabels{i}), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontSize', 10, 'Color', 'red');
            axis([0 1 0 1]); axis off;
            title(plotLabels{i}, 'FontSize', 12, 'Interpreter', 'none', 'Color', 'red');
        end
    end

    sgtitle('DOUT Toolset Overview', 'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');
    panelPath = fullfile(outputDir, sprintf('PANEL_%s.png', upper(p.Results.Prefix)));
    exportgraphics(fig, panelPath, 'Resolution', 300);
    close(fig);

    status.success = true;
    status.panel_path = panelPath;
    fprintf(' ✓ → [%s]\n', panelPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Panel: %s', ME.message);
end
end
