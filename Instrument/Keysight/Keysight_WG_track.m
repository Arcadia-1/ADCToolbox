function Keysight_WG_track(keysight, ch_master, mode)

    Keysight_openConn(keysight);

    if(ch_master ~= 1 && ch_master ~= 2)
        error('Keysight-WG: Channel must be 1 or 2');
    end
    
    if(mode == 0)
        fprintf(keysight,sprintf('SOUR%d:TRAC OFF',ch_master));
    elseif(mode == 1)
        fprintf(keysight,sprintf('SOUR%d:TRAC ON',ch_master));
    elseif(mode == 2)
        fprintf(keysight,sprintf('SOUR%d:TRAC INV',ch_master));
    else
        error('Keysight-WG: Tracking mode must be 0 (off), 1 (on) or 2 (invert)');
    end
        
    Keysight_waitComm;
    
    fclose(keysight);
    
end