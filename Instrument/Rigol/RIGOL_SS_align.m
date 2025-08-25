function RIGOL_SS_align(rigol)

    RIGOL_modelCheck(rigol, RIGOL_DG4000);
    RIGOL_openConn(rigol);
    
    fprintf(rigol,':PHASe:INITiate');
    RIGOL_waitComm;
    
    fclose(rigol);

end