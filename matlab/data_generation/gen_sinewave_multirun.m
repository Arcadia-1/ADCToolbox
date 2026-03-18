%% Centralized Configuration for Sinewave Generation 
common_gen_sinewave;

%% Sinewave with multirun
N_run_list = 16;

% HD to linear amplitude ratio (Harmonic Amp / Fundamental Amp)
hd2_amp = 10^(-100 / 20);
hd3_amp = 10^(-90 / 20);

coef2 = (hd2_amp / (A / 2));
coef3 = (hd3_amp / (A^2 / 4));

sinewave = A * sin(ideal_phase); % Base sinewave (zero mean)
sinewave = sinewave + coef2 * (sinewave.^2) + coef3 * (sinewave.^3);

for k = 1:length(N_run_list)
    N_run = N_run_list(k);
    data_batch = zeros(N, N_run);

    for iter_run = 1:N_run
        data_batch(:, iter_run) = sinewave + DC + randn(N, 1) * Noise_rms;
    end

    filename = fullfile(subFolder, sprintf("batch_sinewave_Nrun_%d.csv", N_run));
    ENoB = plotspec(data_batch',"isplot",0);
    writematrix(data_batch, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end