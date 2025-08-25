function Keysight_WG_turn(keysight, ch1, ch2)

    Keysight_openConn(keysight);
    
    if(ch1 == 1)
        fprintf(keysight,':OUTPut1 ON');
    elseif(ch1 == 0)
        fprintf(keysight,':OUTPut1 OFF');
    end
    Keysight_waitComm;
    
    if(ch2 == 1)
        fprintf(keysight,':OUTPut2 ON');
    elseif(ch2 == 0)
        fprintf(keysight,':OUTPut2 OFF');
    end
    Keysight_waitComm;
    
    fclose(keysight);

end