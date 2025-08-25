function Keysight_WG_setSquare(keysight, ch, freq, high, low, phase, dutycycle)

    Keysight_openConn(keysight);
    
    if(ch ~= 1 && ch ~= 2)
        error('Keysight-WG: Channel must be 1 or 2');
    end
    if(freq < 10^-3 || freq > 200*10^6)
        error('Keysight-WG: Square frequency setting out of range (1m~200M)');
    end
    if(high < low)
        error('Keysight-WG: Square high level must be higher than low level');
    end
    if(dutycycle < 0 || dutycycle > 1)
        error('Keysight-WG: Square duty cycle out of range (0.2~0.8)');
    end

    fprintf(keysight,sprintf('SOUR%d:FUNC SQU',ch));
    fprintf(keysight,sprintf('SOUR%d:FREQ  %e', ch, freq));
    fprintf(keysight,sprintf('SOUR%d:VOLT:HIGH %e V', ch, high));
    fprintf(keysight,sprintf('SOUR%d:VOLT:LOW %e V', ch, low));
    fprintf(keysight,sprintf('SOUR%d:FUNC:SQU:DCYC %e', ch, dutycycle*100));
    fprintf(keysight,sprintf('SOUR%d:PHAS %e', ch,  mod(phase,360))); 
    Keysight_waitComm;

    fclose(keysight);

end