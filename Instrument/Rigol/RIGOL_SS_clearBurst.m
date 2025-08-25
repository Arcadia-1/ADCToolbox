function RIGOL_SS_clearBurst(rigol, ch)

    RIGOL_modelCheck(rigol, RIGOL_DG4000);
    RIGOL_openConn(rigol);
    
    fprintf(rigol,sprintf(':SOURce%d:BURSt:STATe OFF',ch));
    RIGOL_waitComm;
    
    fclose(rigol);

end