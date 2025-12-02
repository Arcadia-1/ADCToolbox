%% Generate sinewave with Dynamic Nonlinearity (Incomplete Settling)
close all; clear; clc; warning("off");
rng(42);
subFolder = fullfile("dataset", "sinewave");
if ~exist(subFolder, 'dir'), mkdir(subFolder); end

%% Simulation Parameters
% Note: Dynamic nonlinearity is frequency dependent.
N = 2^13;
Fs = 1e9; % 1 GSps

% S/H Circuit Constraints
T_track = (1 / Fs) * 0.2; % 0.45ns tracking window

%% Sweep Parameters to generate different distortion levels
% List of Input Frequencies (100 MHz to 1000 MHz in steps of 100 MHz)
Fin_Target_list = 450e6;

% List of Nominal Time Constants (controls linear bandwidth/phase lag)
Tau0_list = 40e-12;

% List of Nonlinear Coefficients (controls dynamic HD3 severity)
% Higher value = More resistance variation = Worse HD3
Coeff_K_list = 0.15;

fs_mhz = Fs / 1e6;
fs_str = sprintf("fs_%gM", fs_mhz);
fs_str = replace(fs_str, '.', 'P');

for k_fin = 1:length(Fin_Target_list)
    Fin_Target = Fin_Target_list(k_fin);

    % Coherent Frequency Calculation (Inline logic replacing findBin)
    J = round(Fin_Target/Fs*N);
    if mod(J, 2) == 0, J = J + 1; end % Ensure odd bin
    Fin = J * Fs / N;

    % Base Sinewave Input (Ideal Voltage)
    t = (0:N - 1) / Fs;
    A = 0.499;
    sinewave = A * sin(2*pi*Fin*t);

    fin_mhz = Fin / 1e6;
    fin_str = sprintf("fin_%.0fM", fin_mhz);
    fin_str = replace(fin_str, '.', 'P');

    for k = 1:length(Tau0_list)
        for k2 = 1:length(Coeff_K_list)

            tau_nom = Tau0_list(k);
            coeff_k = Coeff_K_list(k2);

        % ---- Dynamic Settling Simulation (Memory Effect) ----
        vout = zeros(1, N);
        v_prev = 0; % Initial memory (capacitor charge)

        for n = 1:N
            v_target = sinewave(n);

            % 1. Dynamic Time Constant: Tau changes with Voltage^2
            %    This creates the dynamic nonlinearity (HD3)
            tau_dynamic = tau_nom * (1 + coeff_k * v_target^2);

            % 2. Incomplete Settling Physics
            %    Output depends on WHERE we started (v_prev) -> Memory
            vout(n) = v_target + (v_prev - v_target) * exp(-T_track/tau_dynamic);

            % 3. Update Memory
            v_prev = vout(n);
        end

        data = vout + 0.5 + randn(1, N) * 50e-6;

        tau_str = sprintf("Tau_%dps", round(tau_nom*1e12));
        k_str = replace(sprintf("k_%.4f", coeff_k), ".", "P");

        filename = fullfile(subFolder, sprintf("sinewave_nonlin_dynamic_%s_%s_%s_%s.csv", fs_str, fin_str, tau_str, k_str));
        writematrix(data, filename)

            ENoB = specPlot(data, "isplot", 0);
            fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
        end
    end
end
