function status = toolset_aout_panel(outputDir, varargin)
%TOOLSET_AOUT_PANEL Gather 9 AOUT toolset plots into a 3x3 panel figure
%
%   status = toolset_aout_panel(outputDir)
%   status = toolset_aout_panel(outputDir, 'Prefix', 'aout')
%   status = toolset_aout_panel(outputDir, 'PlotFiles', plotFiles)
%
% Inputs:
%   outputDir  - Directory containing the plot files
%
% Optional Parameters:
%   'Visible'   - Show figure (default: false)
%   'Prefix'    - Filename prefix (default: 'aout')
%   'PlotFiles' - Cell array of 9 PNG file paths (auto-detected if not provided)
%
% Outputs:
%   status - Struct with fields:
%            .success (true if panel created successfully)
%            .panel_path (path to panel figure)
%            .errors (cell array of error messages)
%
% Example:
%   % Auto-detect plot files
%   panel_status = toolset_aout_panel('output/test1', 'Prefix', 'aout');
%
%   % Or provide explicit file paths
%   panel_status = toolset_aout_panel('output/test1', 'PlotFiles', plot_files);

p = inputParser;
addParameter(p, 'Visible', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'Prefix', 'aout', @ischar);
addParameter(p, 'PlotFiles', {}, @iscell);
parse(p, varargin{:});

status = struct('success', false, 'panel_path', '', 'errors', {{}});

% Auto-construct plot file paths if not provided
if isempty(p.Results.PlotFiles)
    plotFiles = {
        fullfile(outputDir, sprintf('%s_1_tomdec.png', p.Results.Prefix));
        fullfile(outputDir, sprintf('%s_2_plotspec.png', p.Results.Prefix));
        fullfile(outputDir, sprintf('%s_3_plotphase.png', p.Results.Prefix));
        fullfile(outputDir, sprintf('%s_4_errsin_code.png', p.Results.Prefix));
        fullfile(outputDir, sprintf('%s_5_errsin_phase.png', p.Results.Prefix));
        fullfile(outputDir, sprintf('%s_6_errPDF.png', p.Results.Prefix));
        fullfile(outputDir, sprintf('%s_7_errAutoCorrelation.png', p.Results.Prefix));
        fullfile(outputDir, sprintf('%s_8_errSpectrum.png', p.Results.Prefix));
        fullfile(outputDir, sprintf('%s_9_errEnvelopeSpectrum.png', p.Results.Prefix));
    };
else
    plotFiles = p.Results.PlotFiles;
end

if length(plotFiles) ~= 9
    error('[Panel generation failed: Expected 9 plot files, got %d]', length(plotFiles));
end

fprintf('[Panel]');
try
    plotLabels = {'(1) tomDecomp'; '(2) specPlot'; '(3) specPlotPhase'; ...
        '(4) errHistSine (code)'; '(5) errHistSine (phase)'; '(6) errPDF'; ...
        '(7) errAutoCorrelation'; '(8) errSpectrum'; '(9) errEnvelopeSpectrum'};

    visOpts = {'off', 'on'};
    fig = figure('Position', [10 10 1800 1000], 'Visible', visOpts{p.Results.Visible+1});
    tlo = tiledlayout(fig, 3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    for i = 1:9
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

    sgtitle('AOUT Toolset Overview', 'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');
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
