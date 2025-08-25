function apx = APX_555B_init()
    apiRoot = 'D:\Program Files\Audio Precision\APx500 6.1\API\';
    NET.addAssembly([apiRoot 'AudioPrecision.API2.dll']);
    apx = AudioPrecision.API.APx500_Application;
    
    apx.Visible = 1;
    apx.OperatingMode = AudioPrecision.API.APxOperatingMode.BenchMode;
    apx.SignalMonitorsEnabled = 0;

    apx.BenchMode.Setup.AnalogOutput.ChannelCount = 1;
    apx.BenchMode.Setup.OutputConnector.Type = AudioPrecision.API.OutputConnectorType.AnalogBalancedAdcTest;
    apx.BenchMode.Setup.AdcTest.VBias = 0;
    apx.BenchMode.Setup.AdcTest.VBiasEnabled = 1;
    apx.BenchMode.Setup.AdcTest.VBiasAutoOn = 1;
    apx.BenchMode.Setup.AdcTest.PinVoltageProtection = 0;
%     apx.BenchMode.Setup.Clocks.TimebaseReference = AudioPrecision.API.TimebaseReferenceType.BNC;
    apx.BenchMode.Setup.AnalogOutput.CommonModeConfiguration = AudioPrecision.API.AnalogBalancedOutputConfigurationType.Normal;
    
%     apx.BenchMode.Generator.AnalogSineMode = AudioPrecision.API.AnalogSineGeneratorMode.HighPerformanceSineGenerator;
%     apx.BenchMode.Generator.PrecisionTune = 1;
    
    apx.BenchMode.Generator.AnalogSineMode = AudioPrecision.API.AnalogSineGeneratorMode.DacGenerator;

end
