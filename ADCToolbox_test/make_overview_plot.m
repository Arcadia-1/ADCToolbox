% Script: make_overview_plot.m
% This script expects the following variables already in workspace:
% SavedImageFiles   cell array of 5 image paths
% OVERVIEW_LABELS   cell array of 5 subplot titles
% output_dir        output folder
% name_without_ext  base name for final overview image

OVERVIEW_LABELS = {; ...
    '(a) Time-domain Error Decomposition', ...
    '(b) Frequency Spectrum (dBFS)', ...
    '(c) Phase-domain Error (dB)', ...
    '(d) Error Histogram by Code (LSB)', ...
    '(e) Error Histogram by Phase (deg)'; ...
    };



fig = figure('Position', [100 100 2000 1200]);   % 更大画布，提升清晰度

% --- Create a 2x3 equal-division layout ---
tlo = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- Place the five generated images into layout ---
for p = 1:5
    nexttile(tlo, p); % 等效于 subplot，但区域更大
    img_path = SavedImageFiles{p};

    try
        img_data = imread(img_path);
        imshow(img_data, 'Border', 'tight');
        title(OVERVIEW_LABELS{p}, 'Interpreter', 'none');
    catch
        text(0.5, 0.5, sprintf('Image Missing: %s', img_path), ...
            'Interpreter', 'none', 'HorizontalAlignment', 'center');
        axis off;
        fprintf('Warning: Could not load %s\n', img_path);
    end
end

% --- Sixth placeholder slot ---
nexttile(tlo, 6);
text(0.5, 0.5, '(f) Placeholder (DNL / Code Overflow)', ...
    'HorizontalAlignment', 'center', 'FontSize', 12);
axis off;

% --- Save overview image ---
output_filepath = fullfile(output_dir, sprintf('OVERVIEW_%s.png', name_without_ext));
exportgraphics(gcf, output_filepath, 'Resolution', 600);
fprintf('[Saved image] -> [%s]\n\n', output_filepath);
