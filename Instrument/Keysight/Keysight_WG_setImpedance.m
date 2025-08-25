function Keysight_WG_setImpedance(keysight, imp_ch1, imp_ch2)

% imp = 1~10K, 0=high-z

    if(imp_ch1 > 10000 || imp_ch1 < 0 || imp_ch2 > 10000 || imp_ch2 < 0 )
        error('Keysight-WG: Output impedance out of range (1~10K, 0 for high-Z)');
    end

    Keysight_openConn(keysight);
    
    if(imp_ch1 == 0)
        fprintf(keysight,'OUTP1:LOAD INF');
    else
        fprintf(keysight,'OUTP1:LOAD %d',floor(imp_ch1));
    end
    Keysight_waitComm;
    
    if(imp_ch2 == 0)
        fprintf(keysight,'OUTP2:LOAD INF');
    else
        fprintf(keysight,'OUTP2:LOAD %d',floor(imp_ch2));
    end
    Keysight_waitComm;

    fclose(keysight);

end