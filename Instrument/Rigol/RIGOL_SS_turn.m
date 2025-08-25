function RIGOL_SS_turn(rigol, ch1, ch2)

    RIGOL_modelCheck(rigol, RIGOL_DG4000);
    RIGOL_openConn(rigol);
    
    if(ch1 == 1)
        fprintf(rigol,':OUTPut1 ON');
    elseif(ch1 == 0)
        fprintf(rigol,':OUTPut1 OFF');
    end
    RIGOL_waitComm;
    
    if(ch2 == 1)
        fprintf(rigol,':OUTPut2 ON');
    elseif(ch2 == 0)
        fprintf(rigol,':OUTPut2 OFF');
    end
    RIGOL_waitComm;
    
    fclose(rigol);

end