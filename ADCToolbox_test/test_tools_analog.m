close all; clear; clc;

input_dir = "ADCToolbox_example_data";
base_output_dir = "ADCToolbox_example_output";

% --- File List Setup (Simplified) ---
manual_file_list = {; ...
    'sinewave_jitter_100fs.csv'; ...
    % "sinewave_noise_1mV.csv";
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
        disp("Error: No files matching '"+search_pattern+"' found in "+input_dir);
        return;
    end
end

% --- Define Core Lists ---
PLOT_PREFIXES = {; ...
    'specPlot_of_', ...
    'specPlotPhase_of_', ...
    'tomDecomp_of_', ...
    'errHistSine_code_of_', ...
    'errHistSine_phase_of_'; ...
    };


% --- Process Each File (Outer Loop) ---
for k = 1:length(file_list_struct)
    filename = file_list_struct(k).name;
    [~, name_without_ext, ~] = fileparts(filename);

    output_dir = fullfile(base_output_dir, name_without_ext);
    if ~exist(output_dir, "dir")
        mkdir(output_dir);
    end

    data = readmatrix(fullfile(input_dir, filename));

    % --- Frequency Estimation (Parameters for tools) ---
    N = length(data);
    freq_est = findFin(data);
    J = findBin(1, freq_est, N);
    Fin_Fs = J / N;

    disp("Processing: "+filename);

    % --- Storage for Saved File Paths (for overview stitching) ---
    SavedImageFiles = {};

    % --- 1. SEQUENTIAL PLOTTING AND SAVING LOOP ---
    for idx = 1:length(PLOT_PREFIXES)

        prefix = PLOT_PREFIXES{idx};

        figure('Position', [100, 100, 600, 400]);

        % --- Execute Function using switch/case ---
        switch idx
            case 1 % tomDecomp
                tomDecomp(data, Fin_Fs, 50, 1);
            case 2 % specPlot
                specPlot(data, 'label', 1, 'harmonic', 0, 'OSR', 1, 'coAvg', 0);
            case 3 % specPlotPhase
                specPlotPhase(data, 'harmonic', 50);
            case 4 % errHistSine (Code)
                errHistSine(data, 1000, Fin_Fs, 'disp', 1, 'mode', 1);
            case 5 % errHistSine (Phase)
                errHistSine(data, 99, Fin_Fs, 'disp', 1, 'mode', 0);
        end

        % Construct and save the file (using char array concatenation is fine here)
        outfile_name = sprintf('%s%s.png', prefix, name_without_ext);
        output_filepath = fullfile(output_dir, outfile_name);
        exportgraphics(gcf, output_filepath, 'Resolution', 300);

        close(gcf);

        % Store the guaranteed file path
        SavedImageFiles{idx} = output_filepath;
    end

end
