function RIGOL_SS_confBurstExt(rigol, ch, edge, tdel, ncycle)

% trigger by external source
% edge = 0: trigger at positive edge, edge = 1: trigger at negative edge
% tdel = delay from trigger to first waveform

    if(ch ~= 1 && ch ~= 2)
        error('RIGOL-SS: Channel must be 1 or 2');
    end
    if(edge ~= 0 && edge ~= 1)
        error('RIGOL-SS: Burst trigger edge must be 0(posedge) or 1(negedge)')
    end
    if(tdel < 0)
        error('RIGOL-SS: Delay must be >=0');
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
    
    fprintf(rigol,sprintf(':SOURce%d:BURSt:TRIGger:SOURce EXTernal',ch));
    RIGOL_waitComm;
    
    if(edge)
        fprintf(rigol,sprintf(':SOURce%d:BURSt:TRIGger:SLOPe NEGative',ch));
    else
        fprintf(rigol,sprintf(':SOURce%d:BURSt:TRIGger:SLOPe POSitive',ch));
    end
    RIGOL_waitComm;
    
    fprintf(rigol,sprintf(':SOURce%d:BURSt:TDELay %e',ch, tdel));
    RIGOL_waitComm;
    
    fprintf(rigol,sprintf(':SOURce%d:BURSt:NCYCles %d',ch, floor(ncycle)));
    RIGOL_waitComm;
    
    fclose(rigol);
end