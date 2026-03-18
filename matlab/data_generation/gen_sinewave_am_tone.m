%% Centralized Configuration for Sinewave Generation 
common_gen_sinewave;

%% Sinewave with True Amplitude Modulation (AM Tone)
am_strength_list = [0.0007]; % modulation depth (m)
am_freq = 99e6; % AM modulation frequency (Hz)

t = (0:N - 1)' / Fs;

for k = 1:length(am_strength_list)
    am_strength = am_strength_list(k);

    sine_zero_mean = A * sin(ideal_phase); % baseband sine

    am_factor = 1 + am_strength * sin(2*pi*am_freq*t); % AM(t) = 1 + m*sin(Ωt)

    data = sine_zero_mean .* am_factor + DC + randn(N, 1) * Noise_rms;

    astr = replace(sprintf('%.4f', am_strength), '.', 'P');
    filename = fullfile(subFolder, sprintf("sinewave_amplitude_modulation_%s.csv", astr));
    ENoB = plotspec(data,"isplot",0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
