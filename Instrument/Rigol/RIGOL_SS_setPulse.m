function RIGOL_SS_setPulse(rigol, ch, high, low, tperiod, tdel, twidth, trise, tfall)

    RIGOL_modelCheck(rigol, RIGOL_DG4000);
    RIGOL_openConn(rigol);
    
    if(ch ~= 1 && ch ~= 2)
        error('RIGOL-SS: Channel must be 1 or 2');
    end
    if(high < low)
        error('RIGOL-SS: Pulse high level must be higher than low level');
    end
    if(tperiod < 20*10^-9)
        error('RIGOL-SS: Pulse period setting out of range (>20ns)');
    end
    if(tdel + twidth + trise + tfall > tperiod)
        error('RIGOL-SS: Pulse timing setting error (t_del+t_width+t_rise+t_fall <= tperiod)');
    end
    if(trise <= 0 || tfall <= 0 || twidth <= 0)
        error('RIGOL-SS: Pulse timing setting error (t_width, t_rise, t_fall >0)');
    end

    fprintf(rigol,sprintf(':SOURce%d:APPLy:PULSe %e,%e,%e,%e',ch,1/tperiod,high-low,(high+low)/2,(1-tdel/tperiod)*360));
    RIGOL_waitComm;

    fprintf(rigol,sprintf(':SOURce%d:PULSe:TRANsition:LEADing %e',ch,trise));
    RIGOL_waitComm;
    
    fprintf(rigol,sprintf(':SOURce%d:PULSe:TRANsition:TRAiling %e',ch,tfall));
    RIGOL_waitComm;
    
    fprintf(rigol,sprintf(':SOURce%d:PULSe:WIDTh %e',ch,twidth+0.5*trise+0.5*tfall));
    RIGOL_waitComm;
    
    fclose(rigol);

end