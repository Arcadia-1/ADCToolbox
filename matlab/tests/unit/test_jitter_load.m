close all; clear; clc; warning("off")

%% Test jitter analysis with deterministic data
% Input: test_data/jitter_sweep/
% Output: test_output/jitter_sweep/

%% Configuration
addpath('matlab/aout');
addpath('matlab/common');
addpath('matlab/test/unit');

inputDir = fullfile('test_data', 'jitter_sweep');
outputDir = fullfile('test_output', 'jitter_sweep');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

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

fprintf('=== test_jitter_load.m ===\n');
fprintf('[Input dir] %s\n', inputDir);
fprintf('[Output dir] %s\n', outputDir);
fprintf('[Fs from config] %.3e Hz\n', Fs_expected);
fprintf('[N from config] %d\n\n', N_expected);
%% Analyze each frequency
for i_freq = 1:length(Fin_list_nominal)

    Fin_nominal = Fin_list_nominal(i_freq);
    fprintf('[Analyzing] Nominal Fin = %d MHz\n', round(Fin_nominal/1e6));

    meas_jitter = zeros(length(Tj_list), 1);
    meas_SNDR = zeros(length(Tj_list), 1);
    set_jitter = zeros(length(Tj_list), 1);
    actual_Fin = zeros(length(Tj_list), 1);
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

        [data_fit, f_norm, mag, dc, phi] = sineFit(read_data);
        Fin_fit = f_norm * Fs_expected;
        actual_Fin(i_tj) = Fin_fit;

        [emean, erms, phase_code, anoi, pnoi] = errHistSine(read_data, 99, f_norm, 0);
        pnoi_array(i_tj) = pnoi;
        anoi_array(i_tj) = anoi;

        jitter_rms = pnoi / (2 * pi * Fin_fit);
        meas_jitter(i_tj) = jitter_rms;

        [ENoB, SNDR, SFDR, SNR, THD, pwr, NF, h] = specPlot(read_data, ...
            'label', 1, 'harmonic', 0, 'winType', @hann, ...
            'OSR', 1, 'coAvg', 0, "isPlot", 0);
        meas_SNDR(i_tj) = SNDR;

        if mod(i_tj, 5) == 0
            fprintf('  [Tj idx %02d/%02d] Fin=%.2fMHz, Set Tj=%.2f fs -> Meas=%.2f fs, [pnoi]=%.3e rad, [SNDR]=%.2f dB\n', ...
                i_tj, length(Tj_list), Fin_fit/1e6, Tj*1e15, jitter_rms*1e15, pnoi, SNDR);
        end

    end

    fprintf('\n');

    Fin_actual_mean = mean(actual_Fin);
    fprintf('[Extracted Fin from data] = %.6f MHz (nominal was %.6f MHz)\n', ...
        Fin_actual_mean/1e6, Fin_nominal/1e6);
    %% Plot results
    figure('Position', [100, 100, 1000, 600]);

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
    title(sprintf("Jitter Analysis (Fin = %.1f MHz)", Fin_actual_mean/1e6), "FontSize", 20);
    legend("Location", "southeast", "FontSize", 16);
    grid on;
    set(gca, "FontSize", 16);

    output_filename = sprintf('jitter_analysis_Fin_%dMHz_matlab.png', round(Fin_nominal/1e6));
    output_filepath = fullfile(outputDir, output_filename);
    saveas(gcf, output_filepath);
    fprintf('[Saved plot] -> [%s]\n\n', output_filepath);
    %% Save results to CSV
    results_filename = sprintf('jitter_results_Fin_%dMHz_matlab.csv', round(Fin_nominal/1e6));
    results_filepath = fullfile(outputDir, results_filename);

    fid = fopen(results_filepath, 'w');
    fprintf(fid, 'Tj_idx,set_jitter_s,set_jitter_fs,meas_jitter_s,meas_jitter_fs,pnoi_rad,anoi,SNDR_dB,actual_Fin_Hz\n');
    for i = 1:length(Tj_list)
        fprintf(fid, '%d,%.12e,%.6f,%.12e,%.6f,%.12e,%.12e,%.2f,%.6e\n', ...
            i, set_jitter(i), set_jitter(i)*1e15, ...
            meas_jitter(i), meas_jitter(i)*1e15, ...
            pnoi_array(i), anoi_array(i), meas_SNDR(i), actual_Fin(i));
    end
    fclose(fid);
    fprintf('[Saved results] -> [%s]\n\n', results_filepath);

end

fprintf('[test_jitter_load COMPLETE]\n');
