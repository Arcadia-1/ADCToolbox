function Keysight_WG_setClkSrc(keysight,int_ext)

% int_ext = 0: internal,  int_ext = 1: external

    Keysight_openConn(keysight);
    
    if(int_ext)
        fprintf(keysight,'ROSC:SOUR EXT');
    else
        fprintf(keysight,'ROSC:SOUR INT');
    end
    Keysight_waitComm;

    fclose(keysight);
end