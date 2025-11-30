close all; clear; clc; warning("off")

%% Generate deterministic jitter sweep test data
% Output: CSV files in test_data/jitter_sweep/

%% Constants
N = 2^12;
Fs = 20e9;
A = 0.49;
offset = 0.5;
amp_noise = 0.00001;

%% Test parameters
Fin_list = [400e6, 900e6, 9000e6];
Tj_list = logspace(-15, -12, 30);

%% Output directory
outputdir = fullfile('dataset', 'others', 'jitter_sweep');
if ~exist(outputdir, 'dir')
    mkdir(outputdir);
end

%% Generate data
fprintf('[Generating jitter sweep test data]\n');
fprintf('[Frequencies] = [%s] MHz\n', sprintf('%d ', Fin_list/1e6));
fprintf('[Jitter range] = %.2f fs to %.2f ps\n', Tj_list(1)*1e15, Tj_list(end)*1e12);
fprintf('[Output dir] = %s\n\n', outputdir);

rng(42, 'twister');

for i_freq = 1:length(Fin_list)

    Fin = Fin_list(i_freq);
    J = findBin(Fs, Fin, N);
    Fin_actual = J/N * Fs;

    fprintf('[Fin] = %d MHz (J=%d, actual=%.6f MHz)\n', ...
        round(Fin/1e6), J, Fin_actual/1e6);

    for i_tj = 1:length(Tj_list)

        Tj = Tj_list(i_tj);
        Ts = 1 / Fs;
        theta = 2 * pi * Fin_actual * (0:N - 1) * Ts;
        phase_noise_rms = 2 * pi * Fin_actual * Tj;
        phase_jitter = randn(1, N) * phase_noise_rms;
        amplitude_noise = randn(1, N) * amp_noise;
        data = sin(theta + phase_jitter) * A + offset + amplitude_noise;

        filename = sprintf('jitter_sweep_Fin_%dMHz_Tj_idx_%02d.csv', ...
            round(Fin/1e6), i_tj);
        filepath = fullfile(outputdir, filename);
        dlmwrite(filepath, data, 'precision', '%.12e');

        if mod(i_tj, 5) == 0
            fprintf('  [Tj idx %02d/%02d] Tj=%.2f fs -> [%s]\n', ...
                i_tj, length(Tj_list), Tj*1e15, filename);
        end

    end

    fprintf('\n');

end

%% Save metadata
metadata_filepath = fullfile(outputdir, 'jitter_sweep_metadata.csv');
fid = fopen(metadata_filepath, 'w');
fprintf(fid, 'Tj_idx,Tj_seconds,Tj_femtoseconds\n');
for i = 1:length(Tj_list)
    fprintf(fid, '%d,%.12e,%.6f\n', i, Tj_list(i), Tj_list(i)*1e15);
end
fclose(fid);
fprintf('[Saved metadata] -> [%s]\n', metadata_filepath);

freq_metadata_filepath = fullfile(outputdir, 'frequency_list.csv');
fid = fopen(freq_metadata_filepath, 'w');
fprintf(fid, 'Fin_Hz,Fin_MHz\n');
for i = 1:length(Fin_list)
    fprintf(fid, '%.6e,%d\n', Fin_list(i), round(Fin_list(i)/1e6));
end
fclose(fid);
fprintf('[Saved frequency list] -> [%s]\n', freq_metadata_filepath);

config_metadata_filepath = fullfile(outputdir, 'config.csv');
fid = fopen(config_metadata_filepath, 'w');
fprintf(fid, 'parameter,value\n');
fprintf(fid, 'Fs,%.6e\n', Fs);
fprintf(fid, 'N,%d\n', N);
fprintf(fid, 'A,%.6f\n', A);
fprintf(fid, 'offset,%.6f\n', offset);
fprintf(fid, 'amp_noise,%.6e\n', amp_noise);
fclose(fid);
fprintf('[Saved config] -> [%s]\n\n', config_metadata_filepath);

fprintf('[Data generation complete] total files = %d\n', length(Fin_list) * length(Tj_list));
