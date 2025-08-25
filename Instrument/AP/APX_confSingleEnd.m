function APX_confSingleEnd(apx, single_end)

    if(single_end)
        apx.BenchMode.Setup.AnalogOutput.CommonModeConfiguration = AudioPrecision.API.AnalogBalancedOutputConfigurationType.SingleEnded;
    else
        apx.BenchMode.Setup.AnalogOutput.CommonModeConfiguration = AudioPrecision.API.AnalogBalancedOutputConfigurationType.Normal;
    end

end