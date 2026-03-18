%% Centralized Configuration for Sinewave Generation 
common_gen_sinewave;

%% Sinewave with baseline drift
baseline_list = 0.003; % different drift amplitudes to sweep

for k = 1:length(baseline_list)
    drift_amp = baseline_list(k);

    sig = A * sin(ideal_phase) + DC + randn(N, 1) * Noise_rms;

    % exponential drift:  from 0 → drift_amp
    drift = (1 - exp(-(0:N - 1)'/N*10)) * drift_amp;
    data = sig + drift;

    bstr = replace(sprintf("%.3f", drift_amp), ".", "P");
    filename = fullfile(subFolder, sprintf("sinewave_drift_%s.csv", bstr));
    ENoB = plotspec(data,"isplot",0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
