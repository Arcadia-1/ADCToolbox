function APX_setSineCh1(apx,freq,amp,offset)

% freq is Hz
% amp is Vp
% offset is Volt
    
    apx.BenchMode.Generator.Frequency.Unit = 'Hz';
    apx.BenchMode.Generator.Frequency.Value = freq;
    apx.BenchMode.Generator.Waveform = 'Sine';

    apx.BenchMode.Generator.Levels.Unit = 'Vp';
    apx.BenchMode.Generator.Levels.SetValue(AudioPrecision.API.OutputChannelIndex.Ch1, amp);
    apx.BenchMode.Generator.Levels.SetOffsetValue(AudioPrecision.API.OutputChannelIndex.Ch1, offset);

end