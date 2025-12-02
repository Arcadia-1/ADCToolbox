function status = toolset_aout_panel(plotFiles, outputDir, varargin)
%TOOLSET_AOUT_PANEL Gather 9 AOUT toolset plots into a 3x3 panel figure
%
%   status = toolset_aout_panel(plotFiles, outputDir)
%   status = toolset_aout_panel(plotFiles, outputDir, 'Prefix', 'aout')
%
% Inputs:
%   plotFiles  - Cell array of 9 PNG file paths (from toolset_aout)
%   outputDir  - Directory to save panel figure
%
% Optional Parameters:
%   'Visible' - Show figure (default: false)
%   'Prefix'  - Filename prefix (default: 'aout')
%
% Outputs:
%   status - Struct with fields:
%            .success (true if panel created successfully)
%            .panel_path (path to panel figure)
%            .errors (cell array of error messages)
%
% Example:
%   % First generate individual plots
%   aout_data = readmatrix('sinewave_jitter.csv');
%   status = toolset_aout(aout_data, 'output/test1');
%
%   % Then gather into panel
%   panel_status = toolset_aout_panel(status.plot_files, 'output/test1', 'Prefix', 'aout');

% Parse inputs
p = inputParser;
addParameter(p, 'Visible', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'Prefix', 'aout', @ischar);
parse(p, varargin{:});
opts = p.Results;

% Set figure visibility (accept 0/1 or true/false)
if opts.Visible
    figVis = 'on';
else
    figVis = 'off';
end

% Initialize status
status.success = false;
status.panel_path = '';
status.errors = {};

% Validate inputs
if ~iscell(plotFiles) || length(plotFiles) ~= 9
    error('[Panel generation failed: Expected 9 plot files, got %d]', length(plotFiles));
end

fprintf('[Panel]');
try
    plotLabels = {
        '(1) tomDecomp';
        '(2) specPlot';
        '(3) specPlotPhase';
        '(4) errHistSine (code)';
        '(5) errHistSine (phase)';
        '(6) errPDF';
        '(7) errAutoCorrelation';
        '(8) errSpectrum';
        '(9) errEnvelopeSpectrum';
    };

    fig = figure('Position', [50 50 1800 1400], 'Visible', figVis);
    tlo = tiledlayout(fig, 3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    for p = 1:length(plotFiles)
        nexttile(tlo, p);
        img_path = plotFiles{p};

        if isfile(img_path)
            img = imread(img_path);
            imshow(img, 'Border', 'tight');
            axis tight;
            axis off;
            title(plotLabels{p}, 'FontSize', 12, 'Interpreter', 'none');
        else
            text(0.5, 0.5, sprintf('Missing:\n%s', plotLabels{p}), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontSize', 10, 'Color', 'red');
            axis([0 1 0 1]);
            axis off;
            title(plotLabels{p}, 'FontSize', 12, 'Interpreter', 'none', 'Color', 'red');
        end
    end

    sgtitle('AOUT Toolset Overview', 'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

    panelPath = fullfile(outputDir, sprintf('PANEL_%s.png', upper(opts.Prefix)));
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
