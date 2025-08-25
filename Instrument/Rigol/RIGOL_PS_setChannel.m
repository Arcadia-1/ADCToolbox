function RIGOL_PS_setChannel(rigol, ch, vout, ilim)

    RIGOL_modelCheck(rigol, RIGOL_DP800);
    RIGOL_openConn(rigol);
    
    if(ch ~= 1 && ch ~= 2 && ch ~= 3)
        error('RIGOL-PS: Channel must be 1/2/3');
    end
    if(vout < 0 || vout > 5)      % limit to 5V for protection, hardware support 30V max
        error('RIGOL-PS: Votage setting out of range (0~5V)');
    end
    if(ilim < 0 || ilim > 3)
        error('RIGOL-PS: Current limit setting out of range (0~3A)');
    end
    
    fprintf(rigol,sprintf(':APPLy CH%d,%.3f,%.3f',ch, vout, ilim));
    RIGOL_waitComm;

    fclose(rigol);

end