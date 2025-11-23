% Script: make_overview_plot.m
% Requires in workspace:
% SavedImageFiles (cell of 9 paths)
% output_dir
% name_without_ext
OVERVIEW_LABELS = {
    '(a) Time-domain Error Decomposition', ...
    '(b) Frequency Spectrum (dBFS)', ...
    '(c) Phase-domain Error (dB)', ...
    '(d) Error Histogram by Code (LSB)', ...
    '(e) Error Histogram by Phase (deg)', ...
    '(f) Error PDF by Magnitude (LSB)', ...
    '(g) Error Autocorrelation', ...
    '(h) Error Spectrum', ...
    '(i) Error Envelope Spectrum'
};
fig = figure('Position', [50 50 1600 1600], 'Visible', 'off');

tlo = tiledlayout(fig, 3, 3, ...
    'TileSpacing','compact', ...
    'Padding','compact');

for p = 1:9
    nexttile(tlo, p);
    img_path = SavedImageFiles{p};
    try
        img = imread(img_path);
        imshow(img, 'Border','tight');
        axis tight;                     
        axis off;                       
        title(OVERVIEW_LABELS{p}, 'FontSize', 16, 'Interpreter','none');
    catch
        text(0.5, 0.5, sprintf('Missing: %s', img_path), ...
            'HorizontalAlignment','center');
        axis off;
    end
end
output_filepath = fullfile(output_dir, sprintf('OVERVIEW_%s_maltab.png', name_without_ext));
exportgraphics(fig, output_filepath, 'Resolution', 600);
fprintf('[Saved image] -> [%s]\n\n', output_filepath);