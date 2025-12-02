%% Centralized Configuration for Sinewave Generation
common_gen_sinewave;
%% Sinewave with clipping
clipping_list = 0.0112;

for k = 1:length(clipping_list)
    clip_th = clipping_list(k);

    sig = A * sin(ideal_phase) + DC + randn(N, 1) * Noise_rms;


    data = min(max(sig, clip_th), 1-clip_th); % apply clipping distortion

    str = replace(sprintf("%.3f", clip_th), ".", "P");
    filename = fullfile(subFolder, sprintf("sinewave_clipping_%s.csv", str));
    ENoB = plotspec(data, "isplot", 0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
