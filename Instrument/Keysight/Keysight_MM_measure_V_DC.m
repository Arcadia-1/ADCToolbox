function result = Keysight_MM_measure_V_DC(keysight,varargin)
% This function measures the DC voltage using a Keysight instrument.
% It accepts optional parameters for measurement mode, range, and resolution.
% Depending on the 'mode', it either directly measures the voltage or configures
% the instrument with the specified range and resolution before measurement.
% Author: Zhishuai Zhang
% Date: 2024.8.13

% Create an input parser to handle optional parameters
p = inputParser;

% Add optional parameters with default values
addOptional(p, 'mode', 0, @isnumeric); % Measurement mode (0 for direct measurement)
addOptional(p, 'measure_range', 10, @isnumeric); % Measurement range in volts
addOptional(p, 'resolution', 0.001, @isnumeric); % Measurement resolution in volts

% Parse the input arguments
parse(p, varargin{:});

% Retrieve the parsed values
mode = p.Results.mode;
measure_range = p.Results.measure_range;
resolution = p.Results.resolution;

% Check the measurement mode and send appropriate commands to the instrument
if mode == 0
    % If mode is 0, perform a direct voltage measurement
    response = writeread(keysight, "MEAS:VOLT:DC?");
else
    % If mode is not 0, configure the instrument with the specified range and resolution
    writeline(keysight, sprintf("CONF:VOLT:DC %.0f,%0.4f", measure_range, resolution));
    response = writeread(keysight, "READ?");
end

% Convert the response from the instrument into a numeric value
result = sscanf(response, '%f');
end