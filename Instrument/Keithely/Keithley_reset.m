function Keithley_reset(kei)

    Keithley_openConn(kei);
    
    fprintf(kei,'*RST');
    Keithley_waitComm;
    
    fclose(kei); 

end