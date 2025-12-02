%% Centralized Configuration for Sinewave Generation
common_gen_sinewave;

%% Sinewave with Amplitude Noise (Random Amplitude Noise)
am_noise_list = 0.0005; % AM noise strength (e.g., 0.1% to 1%)

for k = 1:length(am_noise_list)
    am_strength = am_noise_list(k);

    sinewave_zero_mean = A * sin(ideal_phase); % Ideal sinewave (zero mean)

    % Generate AM noise factor: 1 + random_modulation
    AM_factor = 1 + randn(N, 1) * am_strength;

    % Apply amplitude modulation noise: sinewave * am_factor
    % Add DC offset and small additive white noise (for realism)
    data = sinewave_zero_mean .* AM_factor + DC + randn(N, 1) * Noise_rms;

    astr = replace(sprintf("%.3f", am_strength), ".", "P");
    filename = fullfile(subFolder, sprintf("sinewave_amplitude_noise_%s.csv", astr));
    ENoB = plotspec(data, "isplot", 0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
