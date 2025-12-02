%% Centralized Configuration for Sinewave Generation 
common_gen_sinewave;

%% Sinewave with additive noise
noise_list = 200e-6;

ideal_phase = 2 * pi * Fin * (0:N - 1)' * 1 / Fs;

for k = 1:length(noise_list)
    Noise_rms = noise_list(k);

    data = A * sin(ideal_phase) + DC + randn(N, 1) * Noise_rms;

    nstr = sprintf("%duV", round(noise_amp*1e6));
    filename = fullfile(subFolder, sprintf("sinewave_noise_%s.csv", nstr));
    ENoB = plotspec(data,"isplot",0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
