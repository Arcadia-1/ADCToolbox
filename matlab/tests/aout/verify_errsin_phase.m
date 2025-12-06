%% Verification: Monte Carlo (Absolute Error, Simplified) - Verbose Variables
clear; clc; close all; warning("off")

% --- System Configuration ---
sample_count = 2^13; 
sampling_frequency = 800e6; 
signal_amplitude = 0.5; 
dc_offset = 0; 
monte_carlo_trials = 1; % Increase this for statistical averaging

input_frequency = 127/sample_count * sampling_frequency;

fprintf('\n[Benchmark: errsin vs errsin_hp (Samples=%d, Trials=%d)]\n', sample_count, monte_carlo_trials);
fprintf('Unit: uV (Phase Noise converted via Amplitude=%.1fV)\n', signal_amplitude);
fprintf('Error Format: (E: +Delta) where Delta = Measured - Target (uV)\n');
fprintf('%s\n', repmat('=', 1, 105));
fprintf('| %-38s | %-28s | %-28s |\n', 'Experiment Setup', 'Old: errsin (Legacy)', 'New: errsin_hp (High-Prec)');
fprintf('%s\n', repmat('-', 1, 105));

% --- Test Scenarios Definition ---
% Format: {Name, Input_AM(uV), Input_PM(urad), Input_Thermal(uV)}
% Note: PM(uV) = PM(urad) * 0.5V. Inputs adjusted for strict voltage ratios.
test_scenarios = {
    'Only Thermal',     0,      0,      100;    
    'Only Phase Noise', 0,      1000,   0;      % 1000urad -> 500uV
    'Equal AM and PM',  500,    1000,   0;      % 500uV : 500uV (1:1 Voltage Ratio)
    'PM >> AM (100:1)', 10,     2000,   0;      % 10uV : 1000uV (1:100 Voltage Ratio)
    'AM >> PM (100:1)', 1000,   20,     0;      % 1000uV : 10uV (100:1 Voltage Ratio)
};

% --- Main Verification Loop ---
for scenario_idx = 1:size(test_scenarios, 1)
    % 1. Unpack Scenario Parameters
    description_string = test_scenarios{scenario_idx, 1};
    input_am_uv = test_scenarios{scenario_idx, 2};
    input_pm_urad = test_scenarios{scenario_idx, 3};
    input_thermal_uv = test_scenarios{scenario_idx, 4};
    
    % 2. Calculate Target Values (Ground Truth)
    target_pm_converted_uv = input_pm_urad * signal_amplitude; 
    
    % Total target combines specific noise + thermal noise (RSS)
    target_am_total = sqrt(input_am_uv^2 + input_thermal_uv^2);
    target_pm_total = sqrt(target_pm_converted_uv^2 + input_thermal_uv^2);
    
    % 3. Initialize Accumulators for Monte Carlo Averaging
    sum_squared_am_legacy = 0; 
    sum_squared_pm_legacy = 0; 
    sum_squared_am_high_prec = 0; 
    sum_squared_pm_high_prec = 0;
    
    for trial_idx = 1:monte_carlo_trials
        % Generate Time and Phase Vectors
        time_vector = (0:sample_count-1)' / sampling_frequency;
        phase_clean_rad = 2*pi*input_frequency * time_vector;
        
        % Generate Random Noise Vectors
        noise_am_vector = randn(sample_count, 1) * (input_am_uv * 1e-6);
        noise_pm_vector = randn(sample_count, 1) * (input_pm_urad * 1e-6); 
        noise_thermal_vector = randn(sample_count, 1) * (input_thermal_uv * 1e-6);
        
        % Construct Noisy Signal
        signal_noisy = (signal_amplitude + noise_am_vector) .* sin(phase_clean_rad + noise_pm_vector) + dc_offset + noise_thermal_vector;
        
        % --- Execute Methods ---
        % Method A: Legacy Binning (errsin)
        [~, ~, ~, measured_am_legacy, measured_pm_rad_legacy] = errsin(signal_noisy, 'bin', 100, 'fin', input_frequency/sampling_frequency, 'disp', 0);
        
        % Method B: High Precision Regression (errsin_hp)
        [~, ~, ~, measured_am_hp, measured_pm_rad_hp] = errsin_hp(signal_noisy, 'bin', 100, 'fin', input_frequency/sampling_frequency, 'disp', 0);
        
        % Handle potential NaNs (Safety check)
        if isnan(measured_am_legacy), measured_am_legacy = 0; end
        if isnan(measured_pm_rad_legacy), measured_pm_rad_legacy = 0; end
        if isnan(measured_am_hp), measured_am_hp = 0; end
        if isnan(measured_pm_rad_hp), measured_pm_rad_hp = 0; end
        
        % Accumulate Power (Sum of Squares)
        sum_squared_am_legacy = sum_squared_am_legacy + measured_am_legacy^2;
        sum_squared_pm_legacy = sum_squared_pm_legacy + (measured_pm_rad_legacy * signal_amplitude)^2;
        
        sum_squared_am_high_prec = sum_squared_am_high_prec + measured_am_hp^2;
        sum_squared_pm_high_prec = sum_squared_pm_high_prec + (measured_pm_rad_hp * signal_amplitude)^2;
    end
    
    % 4. Calculate Final RMS Averages
    average_am_legacy = sqrt(sum_squared_am_legacy / monte_carlo_trials);
    average_pm_legacy = sqrt(sum_squared_pm_legacy / monte_carlo_trials);
    
    average_am_high_prec = sqrt(sum_squared_am_high_prec / monte_carlo_trials);
    average_pm_high_prec = sqrt(sum_squared_pm_high_prec / monte_carlo_trials);
    
    % 5. Format Output Strings
    if input_thermal_uv > 0
        config_str_main = description_string;
        config_str_detail = sprintf('[In: Th=%duV]', input_thermal_uv);
    else
        config_str_main = description_string;
        if input_am_uv > 0 && input_pm_urad > 0
            config_str_detail = sprintf('[AM=%d, PM=%dur->%d]', input_am_uv, input_pm_urad, fix(target_pm_converted_uv));
        elseif input_pm_urad > 0
            config_str_detail = sprintf('[PM=%dur -> %duV]', input_pm_urad, fix(target_pm_converted_uv));
        else
            config_str_detail = sprintf('[AM=%duV]', input_am_uv);
        end
    end
    
    % Print Results
    fprintf('| %-38s | AM: %-24s | AM: %-24s |\n', ...
        config_str_main, ...
        format_result_string(average_am_legacy, target_am_total), ...
        format_result_string(average_am_high_prec, target_am_total));
        
    fprintf('| %-38s | PM: %-24s | PM: %-24s |\n', ...
        config_str_detail, ...
        format_result_string(average_pm_legacy, target_pm_total), ...
        format_result_string(average_pm_high_prec, target_pm_total));
        
    fprintf('%s\n', repmat('-', 1, 105));
end

% --- Helper Function ---
function output_str = format_result_string(measured_value, target_value)
    measured_uv = measured_value * 1e6; % Convert to microvolts
    target_uv = target_value;           % Already in microvolts (logic-wise)
    
    if target_value == 0
        output_str = sprintf('%6.2f (Noise)', measured_uv);
    else
        delta_error = measured_uv - target_uv;
        output_str = sprintf('%6.2f (E:%+6.2f)', measured_uv, delta_error);
    end
end