function RIGOL_PS_turn(rigol, ch1, ch2, ch3)

    RIGOL_modelCheck(rigol, RIGOL_DP800);
    RIGOL_openConn(rigol);
    
    if(ch1 == 1)
        fprintf(rigol,':OUTPut CH1,ON');
    elseif(ch1 == 0)
        fprintf(rigol,':OUTPut CH1,OFF');
    end
    RIGOL_waitComm;
    
    if(ch2 == 1)
        fprintf(rigol,':OUTPut CH2,ON');
    elseif(ch2 == 0)
        fprintf(rigol,':OUTPut CH2,OFF');
    end
    RIGOL_waitComm;
    
    if(ch3 == 1)
        fprintf(rigol,':OUTPut CH3,ON');
    elseif(ch2 == 0)
        fprintf(rigol,':OUTPut CH3,OFF');
    end
    
    fclose(rigol);

end