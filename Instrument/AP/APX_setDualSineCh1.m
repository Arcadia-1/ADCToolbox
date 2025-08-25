function APX_setDualSineCh1(apx,freq1,freq2,amp,offset)
    
% freq is Hz
% amp is Vp
% offset is Volt

    apx.BenchMode.Generator.Frequency.Unit = 'Hz';
    apx.BenchMode.Generator.Frequency.Value = freq1;
    apx.BenchMode.Generator.Waveform = 'Sine, Dual';
    apx.BenchMode.Generator.SineDual.FrequencyB.Unit = 'Hz';
    apx.BenchMode.Generator.SineDual.FrequencyB.Value = freq2;
    apx.BenchMode.Generator.SineDual.Split = 0;
    
    apx.BenchMode.Generator.Levels.Unit = 'Vp';
    apx.BenchMode.Generator.Levels.SetValue(AudioPrecision.API.OutputChannelIndex.Ch1, amp);
    apx.BenchMode.Generator.Levels.SetOffsetValue(AudioPrecision.API.OutputChannelIndex.Ch1, offset);

end
