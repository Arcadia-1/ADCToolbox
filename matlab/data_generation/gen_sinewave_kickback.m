%% Centralized Configuration for Sinewave Generation
common_gen_sinewave;
%% Sinewave with 2-step quantization kickback
kickback_strength_list = 0.01;

for k = 1:length(kickback_strength_list)
    kb = kickback_strength_list(k);

    sig = A * sin(ideal_phase) + DC + randn(N, 1) * Noise_rms;

    % two-step quantizer
    msb = floor(sig*2^4) / 2^4; % coarse quantizer (4-bit)
    lsb = floor((sig - msb)*2^12) / 2^12; % fine quantizer  (12-bit)

    % apply kickback (previous MSB affects the next residue)
    msb_shifted = [msb(1); msb(1:end-1)]; % delayed MSB (one-step memory)
    data = msb + lsb + kb * msb_shifted; % kickback injection

    ENoB = plotspec(data, "isplot", 0);
    kstr = replace(sprintf("%.3f", kb), ".", "P");
    filename = fullfile(subFolder, sprintf("sinewave_kickback_%s.csv", kstr));
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
    writematrix(data, filename)
end
