%% Centralized Configuration for Sinewave Generation 
common_gen_sinewave;

%% Sinewave with 2-step quantization gain error
gain_error_list = 0.99;

for k = 1:length(gain_error_list)
    g_err = gain_error_list(k); % interstage gain error (e.g. 0.98, 0.99, 1.00, ...)

    sig = A * sin(ideal_phase) + DC + randn(N, 1) * Noise_rms;

    msb = floor(sig*2^4) / 2^4; % coarse quantizer (4-bit)
    lsb = floor((sig - msb)*2^12) / 2^12; % fine quantizer  (12-bit)

    data = msb * g_err + lsb; % apply interstage gain error

    gstr = replace(sprintf("%.4f", g_err), ".", "P");
    filename = fullfile(subFolder, sprintf("sinewave_gain_error_%s.csv", gstr));
    ENoB = plotspec(data,"isplot",0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
