%% Centralized Configuration for Sinewave Generation
common_gen_sinewave;
%% Sinewave with jitter
Tj_list = 300e-15;

for k = 1:length(Tj_list)
    Tj = Tj_list(k);

    phase_noise_rms = 2 * pi * Fin * Tj; % convert jitter(sec) -> phase jitter(rad)
    phase_jitter = randn(N, 1) * phase_noise_rms; % random jitter

    data = A * sin(ideal_phase+phase_jitter) + DC + randn(N, 1) * Noise_rms; % jittered signal

    filename = fullfile(subFolder, sprintf("sinewave_jitter_%dfs.csv", round(Tj*1e15)));
    ENoB = plotspec(data, "isplot", 0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
