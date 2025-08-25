function Keysight_WG_setSine(keysight, ch, freq, amp, offset, phase)

    Keysight_openConn(keysight);
    
    if(ch ~= 1 && ch ~= 2)
        error('Keysight-WG: Channel must be 1 or 2');
    end
    if(freq < 10^-3 || freq > 250*10^6)
        error('Keysight-WG: Sine frequency setting out of range (1m~250M)');
    end
    
    fprintf(keysight,sprintf('SOUR%d:FUNC SIN',ch));
    fprintf(keysight,sprintf('SOUR%d:FREQ %e', ch, freq));
    fprintf(keysight,sprintf('SOUR%d:VOLT %e Vpp', ch,  amp));
    fprintf(keysight,sprintf('SOUR%d:VOLT:OFFS %e V', ch,  offset));
    fprintf(keysight,sprintf('SOUR%d:PHAS %e', ch,  mod(phase,360)));
    Keysight_waitComm;
    
    fclose(keysight);

end