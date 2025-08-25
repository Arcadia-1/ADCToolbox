function RIGOL_SS_confBurstInt(rigol, ch, tperiod, ncycle)

% trigger by external source
% known bug: burst period is not synchronized with the other channel!
%   to synchronized between channels, use external trigger.

    if(ch ~= 1 && ch ~= 2)
        error('RIGOL-SS: Channel must be 1 or 2');
    end
    if(tperiod <= 0)
        error('RIGOL-SS: T-period be >0');
    end
    if(ncycle < 1)
        error('RIGOL-SS: N-Cycle must be >=1');
    end

    RIGOL_modelCheck(rigol, RIGOL_DG4000);
    RIGOL_openConn(rigol);
    
    fprintf(rigol,sprintf(':SOURce%d:BURSt:STATe ON',ch));
    RIGOL_waitComm;
    
    fprintf(rigol,sprintf(':SOURce%d:BURSt:MODE TRIGgered',ch));
    RIGOL_waitComm;
    
    fprintf(rigol,sprintf(':SOURce%d:BURSt:TRIGger:SOURce INTernal',ch));
    RIGOL_waitComm;
    
    fprintf(rigol,sprintf(':SOURce%d:BURSt:INTernal:PERiod %e',ch, tperiod));
    RIGOL_waitComm;
    
    fprintf(rigol,sprintf(':SOURce%d:BURSt:NCYCles %d',ch, floor(ncycle)));
    RIGOL_waitComm;
    
    fclose(rigol);
end