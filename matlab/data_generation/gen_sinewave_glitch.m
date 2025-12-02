%% Centralized Configuration for Sinewave Generation 
common_gen_sinewave;

%% Sinewave with random glitch injection
glitch_prob_list = 0.00015; % 0.1%, 1%, 10%

for k = 1:length(glitch_prob_list)
    gprob = glitch_prob_list(k);

    sig = A * sin(ideal_phase) + DC + randn(N, 1) * Noise_rms;

    % glitch injection: upward spike of +0.1 with probability gprob
    glitch = (rand(N, 1) < gprob) * 0.1;
    data = sig + glitch;

    pstr = replace(sprintf("%.6f", gprob), ".", "P");
    filename = fullfile(subFolder, sprintf("sinewave_glitch_%s.csv", pstr));
    ENoB = plotspec(data,"isplot",0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
