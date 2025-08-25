function RIGOL_reset(rigol)

    RIGOL_openConn(rigol);
    
    fprintf(rigol,'*RST');
    RIGOL_waitComm;
    
    fclose(rigol); 

end