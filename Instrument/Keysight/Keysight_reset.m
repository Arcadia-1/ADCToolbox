function Keysight_reset(keysight)

    Keysight_openConn(keysight);
    
    fprintf(keysight,'*RST');
    Keysight_waitComm;
    
    fclose(keysight); 

end