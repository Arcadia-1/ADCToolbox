function RIGOL_SS_setSine(rigol, ch, freq, amp, offset, phase)

    RIGOL_modelCheck(rigol, RIGOL_DG4000);
    RIGOL_openConn(rigol);
    
    if(ch ~= 1 && ch ~= 2)
        error('RIGOL-SS: Channel must be 1 or 2');
    end
    if(freq < 10^-3 || freq > 200*10^6)
        error('RIGOL-SS: Sine frequency setting out of range (1m~200M)');
    end
    
    fprintf(rigol,sprintf(':SOURce%d:APPLy:SINusoid %e,%e,%e,%e',ch,freq,amp,offset,mod(phase,360)));
    RIGOL_waitComm;

    fclose(rigol);

end