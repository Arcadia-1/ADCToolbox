function RIGOL_SS_setSquare(rigol, ch, freq, high, low, phase, dutycycle)

    RIGOL_modelCheck(rigol, RIGOL_DG4000);
    RIGOL_openConn(rigol);
    
    if(ch ~= 1 && ch ~= 2)
        error('RIGOL-SS: Channel must be 1 or 2');
    end
    if(freq < 10^-3 || freq > 200*10^6)
        error('RIGOL-SS: Square frequency setting out of range (1m~200M)');
    end
    if(high < low)
        error('RIGOL-SS: Square high level must be higher than low level');
    end
    if(dutycycle < 0 || dutycycle > 1)
        error('RIGOL-SS: Square duty cycle out of range (0.2~0.8)');
    end

    fprintf(rigol,sprintf(':SOURce%d:APPLy:SQUare %e,%e,%e,%e',ch,freq,high-low,(high+low)/2,mod(phase,360)));
    RIGOL_waitComm;

    fprintf(rigol,sprintf(':SOURce%d:FUNCtion:SQUare:DCYCle %e',ch,dutycycle*100));
    RIGOL_waitComm;

    fclose(rigol);

end