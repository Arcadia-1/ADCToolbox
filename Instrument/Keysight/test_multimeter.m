% This script periodically measures DC voltage using a Keysight multimeter.
% It captures voltage readings at specified intervals, storing both the voltage values and their corresponding time differences.
% The data is saved to a CSV file in bulk at the end, with headers.
% Author: Zhishuai Zhang
% Date: 2024.8.13

% Initialize Keysight multimeter if not already initialized
if (~exist("keysight","var"))
    keysight = Keysight_MM_init(Keysight_MM_5);
end

% User settings
interval = 1; % Interval between measurements (seconds)
num_measurements = 100; % Number of measurements
csv_filename = 'voltage_data.csv'; % CSV file name

% Preallocate matrix for time difference and voltage data
data_matrix = zeros(num_measurements, 2);

% Measurement loop
for i = 1:num_measurements
    % Measure voltage and get current timestamp
    voltage = Keysight_MM_measure_V_DC(keysight, 1, 1, 0.001);
    timestamp = datetime('now'); % Get current time

    if i == 1
        initial_time = timestamp; % Record the time of the first measurement
    end

    time_difference = seconds(timestamp - initial_time); % Calculate time difference in seconds
    data_matrix(i, :) = [time_difference, voltage]; % Store time difference and voltage
    
    % Print both absolute time and time difference
    fprintf("[Keysight MM] Absolute Time: %s, Time difference: %0.2f seconds, DC voltage = %0.4f V\n", ...
        datestr(timestamp, 'yyyy-mm-dd HH:MM:SS'), time_difference, voltage);

    % Wait for the next measurement
    pause(interval);
end

% Save all data at the end with headers
writematrix(["second", "voltage"; data_matrix], csv_filename); % Save the entire matrix with headers to CSV