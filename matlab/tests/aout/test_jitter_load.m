%% test_jitter_load.m - Unit test for jitter analysis
close all; clear; clc; warning("off")

%% Configuration
verbose = 1;
inputDir = fullfile('test_dataset', 'jitter_sweep');
outputDir = fullfile('test_output', 'jitter_sweep');
figureDir = "test_plots";
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

config_filepath = fullfile(inputDir, 'config.csv');
if ~exist(config_filepath, 'file')
    error('[Config file not found] Please run generate_jitter_sweep_data.m first');
end

config_data = readcell(config_filepath);
Fs_expected = config_data{2, 2};
N_expected = config_data{3, 2};

metadata_filepath = fullfile(inputDir, 'jitter_sweep_metadata.csv');
metadata = readmatrix(metadata_filepath);
Tj_list = metadata(:, 2);

freq_metadata_filepath = fullfile(inputDir, 'frequency_list.csv');
freq_metadata = readmatrix(freq_metadata_filepath);
Fin_list_nominal = freq_metadata(:, 1);

%% Analyze each frequency
for i_freq = 1:length(Fin_list_nominal)
    Fin_nominal = Fin_list_nominal(i_freq);
    fprintf('[%s] [%d/%d] [Fin = %d MHz]\n', mfilename, i_freq, length(Fin_list_nominal), round(Fin_nominal/1e6));

    meas_jitter = zeros(length(Tj_list), 1);
    meas_SNDR = zeros(length(Tj_list), 1);
    set_jitter = zeros(length(Tj_list), 1);
    pnoi_array = zeros(length(Tj_list), 1);
    anoi_array = zeros(length(Tj_list), 1);

    for i_tj = 1:length(Tj_list)
        Tj = Tj_list(i_tj);
        set_jitter(i_tj) = Tj;

        filename = sprintf('jitter_sweep_Fin_%dMHz_Tj_idx_%02d.csv', ...
            round(Fin_nominal/1e6), i_tj);
        filepath = fullfile(inputDir, filename);

        if ~exist(filepath, 'file')
            warning('[File not found] %s', filepath);
            continue;
        end

        read_data = readmatrix(filepath);
        N = length(read_data);

        if N ~= N_expected
            warning('[N mismatch] Data has N=%d, config expects N=%d', N, N_expected);
        end

        f_norm = findfreq(read_data);
        Fin_fit = f_norm * Fs_expected;

        [emean, erms, xx, anoi, pnoi] = errsin(read_data, 'bin', 99, 'fin', f_norm, 'disp', 0, 'xaxis', 'phase');
        pnoi_array(i_tj) = pnoi;
        anoi_array(i_tj) = anoi;

        jitter_rms = pnoi / (2 * pi * Fin_fit);
        meas_jitter(i_tj) = jitter_rms;

        [enob, sndr, sfdr, snr, thd, sigpwr, noi, nsd, h] = plotspec(read_data, ...
            'label', 1, 'harmonic', 0, 'window', @hann, ...
            'OSR', 1, 'averageMode', 0, "disp", 0);
        meas_SNDR(i_tj) = sndr;
    end

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    yyaxis left
    loglog(set_jitter, set_jitter, 'k--', 'LineWidth', 1.5, "DisplayName", "Set jitter");
    hold on;
    loglog(set_jitter, meas_jitter, 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b', "DisplayName", "Calculated jitter");
    ylabel("Jitter (seconds)", "FontSize", 18);
    ylim([min(set_jitter) * 0.5, max(set_jitter) * 2]);

    yyaxis right
    semilogx(set_jitter, meas_SNDR, 's-', 'LineWidth', 2, 'MarkerSize', 8, "DisplayName", "SNDR");
    ylabel("SNDR (dB)", "FontSize", 18);
    ylim([0, 100]);

    xlabel("Set jitter (seconds)", "FontSize", 18);
    title(sprintf("Jitter Analysis (Fin = %.1f MHz)", Fin_fit/1e6), "FontSize", 20);
    legend("Location", "southeast", "FontSize", 16);
    grid on;
    set(gca, "FontSize", 16);

    fin_str = replace(sprintf("%fin_.0fM", Fin_fit/1e6), ".", "P");
    figureName = sprintf("jitter_sweep_%s_matlab.png", fin_str);
    saveFig(figureDir, figureName, verbose);

    saveVariable(outputDir, set_jitter, verbose);
    saveVariable(outputDir, pnoi_array, verbose);
    saveVariable(outputDir, anoi_array, verbose);
    saveVariable(outputDir, meas_jitter, verbose);
    saveVariable(outputDir, meas_SNDR, verbose);

end
