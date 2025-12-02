%% Centralized Configuration for Sinewave Generation
common_gen_sinewave;

%% Sinewave with reference modulation error
ref_error_list = 0.00075; % different reference mismatch levels

for k = 1:length(ref_error_list)
    ref_amp = ref_error_list(k);

    sig = A * sin(ideal_phase) + DC + randn(N, 1) * Noise_rms;

    msb = floor(sig*2^4) / 2^4;
    lsb = floor((sig - msb)*2^12) / 2^12;

    data = msb + lsb + (msb - 0.5) .* ((-1).^(0:N - 1)') * ref_amp;

    rstr = replace(sprintf("%.3f", ref_amp), ".", "P");
    filename = fullfile(subFolder, sprintf("sinewave_ref_error_%s.csv", rstr));
    ENoB = plotspec(data,"isplot",0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
