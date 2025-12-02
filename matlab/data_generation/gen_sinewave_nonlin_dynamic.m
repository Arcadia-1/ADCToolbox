%% Centralized Configuration for Sinewave Generation
common_gen_sinewave;
%% Simulation Parameters
T_track = (1 / Fs) * 0.2; % 0.45ns tracking window

%% Sweep Parameters to generate different distortion levels
% List of Input Frequencies (100 MHz to 1000 MHz in steps of 100 MHz)
Fin_Target_list = 450e6;

% List of Nominal Time Constants (controls linear bandwidth/phase lag)
Tau0_list = 40e-12;

% List of Nonlinear Coefficients (controls dynamic HD3 severity)
Coeff_K_list = 0.15;

fs_str = sprintf("fs_%gM", Fs / 1e6);
fs_str = replace(fs_str, '.', 'P');

for k_fin = 1:length(Fin_Target_list)

    Fin_Target = Fin_Target_list(k_fin);

    % Coherent Frequency Calculation (Inline logic replacing findBin)
    J = round(Fin_Target/Fs*N);
    if mod(J, 2) == 0, J = J + 1; end % Ensure odd bin
    Fin = J * Fs / N;

    sinewave = A * sin(ideal_phase);

    for k = 1:length(Tau0_list)
        for k2 = 1:length(Coeff_K_list)

            tau_nom = Tau0_list(k);
            coeff_k = Coeff_K_list(k2);

            % ---- Dynamic Settling Simulation (Memory Effect) ----
            vout = zeros(N, 1);
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

            data = vout + DC + randn(N, 1) * Noise_rms;

            tau_str = sprintf("Tau_%dps", round(tau_nom*1e12));
            k_str = replace(sprintf("k_%.4f", coeff_k), ".", "P");


            fin_str = sprintf("fin_%.0fM", Fin / 1e6);
            fin_str = replace(fin_str, '.', 'P');
            filename = fullfile(subFolder, sprintf("sinewave_nonlin_dynamic_%s_%s_%s_%s.csv", fs_str, fin_str, tau_str, k_str));
            writematrix(data, filename)

            ENoB = plotspec(data, "isplot", 0);
            fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
        end
    end
end
