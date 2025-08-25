
%% initialize
keysight = Keysight_MM_init(Keysight_MM_5);

%% measurement

% measure DC voltage with default settings
response = writeread(keysight,"MEAS:VOLT:DC?");
floatValue = sscanf(response, '%f');
fprintf("[Keysight MM] DC voltage = %0.4f V\n", floatValue)


% The following example configures the instrument for 2-wire resistance
% measurements, triggers the instru-ment to take a measurement, and reads
% the measurement. The 1 kΩ range is selected with 0.1 Ω res-olution.
response = writeread(keysight, "MEAS:RES? 100, 0.1");

%% display control
text = "Testing! Do not touch!";
writeline(keysight, sprintf("DISP:TEXT '%s'", text));


writeline(keysight, "DISP:TEXT:CLEAR");

writeline(keysight, "DISP:STATE OFF");
writeline(keysight, "DISP:STATE ON");

disp_state = writeread(keysight, "DISP:STAT?")

%% disconnect
% When no references to the same connection exist in other variables, you
% can disconnect the device by clearing the workspace variable.

clear(keysight)