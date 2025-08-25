function RIGOL_SS_setClkSrc(rigol,int_ext)

% int_ext = 0: internal,  int_ext = 1: external

    RIGOL_modelCheck(rigol, RIGOL_DG4000);
    RIGOL_openConn(rigol);
    
    if(int_ext)
        fprintf(rigol,':SYSTem:ROSCillator:SOURce EXTernal');
    else
        fprintf(rigol,':SYSTem:ROSCillator:SOURce INTernal');
    end
    RIGOL_waitComm;

    fclose(rigol);
end