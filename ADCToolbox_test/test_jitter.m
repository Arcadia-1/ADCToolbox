close all ; clear; clc; warning("off")

%% constants
N = 2^16;
Fs = 10e9;
J = findBin(Fs, 1000e6, N);
Fin = J/N * Fs;

A = 0.49;
offset = 0.5;
amp_noise = 0.00001;

%% jitter list
Tj_list = logspace(-15,-12,20);   % you can edit this

%% number of random trials per jitter value
N_random = 5;

%% results
meas_jitter = zeros(length(Tj_list), N_random);
meas_jitter_new = zeros(length(Tj_list), N_random);
meas_SNDR = zeros(length(Tj_list), N_random);

%% loops
for i = 1:length(Tj_list)

    Tj = Tj_list(i);

    for k = 1:N_random

        % ideal phase
        Ts = 1 / Fs;
        theta = 2 * pi * Fin * (0:N - 1) * Ts;

        % convert jitter(sec) -> phase jitter(rad)
        phase_noise_rms = 2 * pi * Fin * Tj;

        % random jitter
        phase_jitter = randn(1, N) * phase_noise_rms;

        % jittered signal
        data = sin(theta + phase_jitter) * A + offset + randn(1, N) * amp_noise;

        % data = sin(theta + phase_jitter) * A .* (1+randn(1, N)*0.02) + offset + randn(1, N) * amp_noise;

        % extract jitter by errHistSine
        [emean, erms, phase_code, anoi, pnoi] = errHistSine(data, 99, J/N, 0);

        amp_err = sqrt(erms.^2 - min(erms)^2);
        norm_amp_err = amp_err * 2 / (max(data) - min(data));
        phase_slope = abs(sin(phase_code/360*2*pi));

        jitter_on_phase = norm_amp_err ./ phase_slope / (2*pi*Fin);
        jitter_on_phase(~isfinite(jitter_on_phase)) = 0;

        jitter_rms = mean(jitter_on_phase(2:end));

        jitter_rms_new = pnoi / (2*pi*Fin);

        meas_jitter(i, k) = jitter_rms;
        meas_jitter_new(i, k) = jitter_rms_new;


        [ENoB,SNDR,SFDR,SNR,THD,pwr,NF,h] = specPlot(data, 'label', 1, 'harmonic', 0, 'winType', @hann, 'OSR', 1, 'coAvg', 0, "isPlot", 0);

        meas_SNDR(i, k) = SNDR;


        fprintf("[Tj]=%8.2ffs, [trial %d] -> [ENoB=%0.2f] [measured jitter] %.2ffs [measured jitter2] %.2ffs\n", ...
            Tj*1e15, k, ENoB, jitter_rms*1e15, jitter_rms_new*1e15);

    end
end

%% compute average of N_random runs
avg_meas_jitter = mean(meas_jitter, 2);
avg_meas_jitter_new = mean(meas_jitter_new, 2);
avg_meas_SNDR = mean(meas_SNDR, 2);

figure;

%% left axis: jitter
yyaxis left
loglog(Tj_list, Tj_list, 'k--', 'LineWidth', 1.5, "DisplayName","set jitter"); hold on;
loglog(Tj_list, avg_meas_jitter, 'ro-', 'LineWidth', 2, 'MarkerSize', 8, "DisplayName","old method");
loglog(Tj_list, avg_meas_jitter_new, 'bs-', 'LineWidth', 2, 'MarkerSize', 8, "DisplayName","new method");

ylabel("Calculated jitter (seconds)", "FontSize", 18);

%% right axis: SNDR
yyaxis right
semilogx(Tj_list, avg_meas_SNDR, 's-', 'LineWidth', 2, 'MarkerSize', 8, "DisplayName","SNDR");
ylabel("SNDR (dB)", "FontSize", 18);
ylim([0,100])

%% shared x axis
xlabel("Set jitter (seconds)", "FontSize", 18);

title(sprintf("Jitter and SNDR (Fin = %0.2fGHz)",Fin/1e9), "FontSize", 20);

legend("Location", "southeast", "FontSize", 16);

grid on;
set(gca, "FontSize", 16);
