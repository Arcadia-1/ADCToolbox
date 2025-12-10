clear; clc; close all; warning('off');

figureDir = "test_plots";
if ~exist(figureDir, 'dir'), mkdir(figureDir); end

N = 2^13;
Fs = 800e6;
Fin = 1/N * Fs;
A = 0.5;
DC = 0;

Vals_AM_V      = [100e-6]; 
Vals_PM_V      = [100e-6]; 
Vals_Thermal_V = [0, 100e-6];
case_ctr = 0;

% --- Nested Loops for Full Sweep ---
for i = 1:length(Vals_AM_V)
    for j = 1:length(Vals_PM_V)
        for k = 1:length(Vals_Thermal_V)
            
            case_ctr = case_ctr + 1;
            
            % Get current parameters
            Target_AM_V = Vals_AM_V(i);
            Target_PM_V = Vals_PM_V(j);
            Target_Thermal_V = Vals_Thermal_V(k);
            
            % Derived parameters
            Target_PM_Rad = Target_PM_V / A;
            
            % Prepare display values (integer uV)
            am_u = round(Target_AM_V * 1e6);
            pm_u = round(Target_PM_V * 1e6);
            th_u = round(Target_Thermal_V * 1e6);
            
            % --- Signal Generation ---
            t = (0:N-1)' / Fs;
            phase_clean = 2*pi*Fin*t;
            
            % Reset RNG for reproducible noise patterns relative to amplitude
            rng(42); 
            n_am = randn(N, 1) * Target_AM_V;
            n_pm = randn(N, 1) * Target_PM_Rad;
            n_th = randn(N, 1) * Target_Thermal_V;
            
            sig_noisy = (A + n_am) .* sin(phase_clean + n_pm) + DC + n_th;
            
            % ---------------------------------------------------------
            % 1. Run Old Function (errsin)
            % ---------------------------------------------------------
            fig_old = figure('Visible', 'off', 'Position', [100, 100, 800, 600]);
            tic;
            [~, ~, ~, meas_am_old, meas_pm_rad_old] = errsin(sig_noisy, 'bin', 100, 'fin', Fin/Fs, 'disp', 1);
            time_old = toc;
            
            sgtitle(sprintf('(Old) errsin: AM=%duV, PM=%duV, White Noise=%duV', am_u, pm_u, th_u));
            
            fname_old = sprintf("Case%d_AM_%du_PM_%du_WN_%du_errsin.png", case_ctr, am_u, pm_u, th_u);
            saveFig(figureDir, fname_old, 0);
            close all;
            
            % ---------------------------------------------------------
            % 2. Run New Function (errsin_hp)
            % ---------------------------------------------------------
            fig_new = figure('Visible', 'off', 'Position', [100, 100, 800, 600]);
            tic;
            [~, ~, ~, meas_am_hp, meas_pm_rad_hp] = errsin_robust(sig_noisy, 'bin', 100, 'fin', Fin/Fs, 'disp', 1);
            time_hp = toc;
            
            sgtitle(sprintf('(Robust): AM=%duV, PM=%duV, White Noise=%duV', am_u, pm_u, th_u));
            
            fname_new = sprintf("Case%d_AM_%du_PM_%du_WN_%du_errsin_robust.png", case_ctr, am_u, pm_u, th_u);
            saveFig(figureDir, fname_new, 0);
            close all;
            
            % ---------------------------------------------------------
            % 3. Console Output Comparison
            % ---------------------------------------------------------
            meas_am_old_v = meas_am_old;
            meas_pm_old_v = meas_pm_rad_old * A;
            
            meas_am_hp_v = meas_am_hp;
            meas_pm_hp_v = meas_pm_rad_hp * A;
            
            fprintf('--------------------------------------------------------------------\n');
            fprintf(' %-20s| %-12s | %-12s | %-12s |\n', 'Metric', 'Input (Ref)', 'errsin (Old)', 'robust (New)');
            fprintf('--------------------------------------------------------------------\n');
            fprintf(' Phase Noise     (uV)| %12.2f | %12.2f | %12.2f |\n', Target_PM_V*1e6, meas_pm_old_v*1e6, meas_pm_hp_v*1e6);
            fprintf(' Amplitude Noise (uV)| %12.2f | %12.2f | %12.2f |\n', Target_AM_V*1e6, meas_am_old_v*1e6, meas_am_hp_v*1e6);
            fprintf(' White Noise     (uV)| %12.2f | %12s | %12s |\n', Target_Thermal_V*1e6, '-','-');
            fprintf(' Time Cost        (s)| %12s | %12.4f | %12.4f |\n', '-', time_old, time_hp);
            fprintf('--------------------------------------------------------------------\n\n');
            
        end
    end
end