close all; clear; clc; warning("off")


performAnalysis = 0;


input_dir = "ADCToolbox_example_data";
base_output_dir = "ADCToolbox_example_output";

% 'sinewave_jitter_100fs.csv'; ...
manual_file_list = { ...
    "sinewave_kickback_0P009.csv"
    };

if ~exist(base_output_dir, "dir")
    mkdir(base_output_dir);
end
if ~isempty(manual_file_list)
    file_list_struct = struct('name', manual_file_list);
else
    search_pattern = "Sinewave_*.csv";
    file_list_struct = dir(fullfile(input_dir, search_pattern));

    if isempty(file_list_struct)
        disp("Error: No files found.");
        return;
    end
end

PLOT_PREFIXES = {; ...
    'tomDecomp_of_', ...
    'specPlot_of_', ...
    'specPlotPhase_of_', ...
    'errHistSine_code_of_', ...
    'errHistSine_phase_of_', ...
    'errPDF_of_', ...
    'errACF_of_', ...
    'errSpectrum_of_', ...
    'errEnvelopeSpectrum_of_'; ...
    };

TotalFiles = length(file_list_struct);
for k = 1:length(file_list_struct)

    filename = file_list_struct(k).name;
    [~, name_without_ext] = fileparts(filename);
    current_progress = (k / TotalFiles) * 100;
    fprintf('[%.1f%%] Processing File %d of %d: %s\n', ...
        current_progress, k, TotalFiles, filename);


    output_dir = fullfile(base_output_dir, name_without_ext);
    if ~exist(output_dir, "dir")
        mkdir(output_dir);
    end

    if performAnalysis == true
        data = readmatrix(fullfile(input_dir, filename));

        N = length(data);
        Fin = findFin(data);
        J = findBin(1, Fin, N);
        relative_Fin = J / N;

        SavedImageFiles = {};

        % Get err_data once
        [data_fit, ~, ~, ~, ~] = sineFit(data);
        err_data = data - data_fit;

        for idx = 1:length(PLOT_PREFIXES)

            prefix = PLOT_PREFIXES{idx};
            figure('Position', [100, 100, 600, 400], 'Visible', 'off');

            switch idx
                case 1 % tomDecomp
                    tomDecomp(data, relative_Fin, 50, 1);

                case 2 % specPlot
                    specPlot(data, 'label', 1, 'harmonic', 0, 'OSR', 1, 'coAvg', 0);


                case 3 % specPlotPhase
                    specPlotPhase(data, 'harmonic', 50);


                case 4 % errHistSine (code)
                    errHistSine(data, 1000, relative_Fin, 'disp', 1, 'mode', 1);

                case 5 % errHistSine (phase)
                    errHistSine(data, 99, relative_Fin, 'disp', 1, 'mode', 0);

                case 6 % errPDF
                    errPDF(err_data);

                case 7 % errAutoCorrelation
                    errAutoCorrelation(err_data, "MaxLag", 300);

                case 8 % Error Spectrum (using specPlot)
                    specPlot(err_data, "label", 0);
                    % title("Error Spectrum")

                case 9 % Error Envelope Spectrum
                    errEnvelopeSpectrum(err_data);
            end

            outfile_name = sprintf('%s%s_matlab.png', prefix, name_without_ext);
            output_filepath = fullfile(output_dir, outfile_name);
            exportgraphics(gcf, output_filepath, 'Resolution', 300);

            close(gcf);

            SavedImageFiles{idx} = output_filepath;
        end
    else
        SavedImageFiles = cell(size(PLOT_PREFIXES));
        for idx = 1:length(PLOT_PREFIXES)
            prefix = PLOT_PREFIXES{idx};
            outfile_name = sprintf('%s%s_matlab.png', prefix, name_without_ext);
            output_filepath = fullfile(output_dir, outfile_name);
            SavedImageFiles{idx} = output_filepath;
        end
    end
    make_overview_plot;
    close all;
end
