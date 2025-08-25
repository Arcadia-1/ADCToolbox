% Add reference to the API wrapper.
apiRoot = 'C:\Program Files (x86)\Audio Precision\APx500 4.4\Api\';
NET.addAssembly([apiRoot 'AudioPrecision.API2.dll'])

% Create an instance of the application, and make it visible (it doesn't
% have to be visible to work, it just lets us see what's happening).
apx = AudioPrecision.API.APx500_Application;
apx.Visible = true;

% Turn off the signal monitors.
apx.SignalMonitorsEnabled = false;

% Select the Frequency Response measurement.
apx.ShowMeasurement('Signal Path1', 'Frequency Response')

% Set the generator parameters. Matlab can only set a property on an 
% explicit handle.
gen = apx.FrequencyResponse.Generator;
startFreq = gen.StartFrequency;
startFreq.Unit = 'Hz';
startFreq.Value = 30;
stopFreq = gen.StopFrequency;
stopFreq.Value = 18000;
levels = gen.Levels;
levels.Unit = 'Vrms';
levels.SetValue(AudioPrecision.API.OutputChannelIndex.Ch1, 1.5);

% Start the measurement. Success is true if the measurement worked.
success = apx.FrequencyResponse.Start;