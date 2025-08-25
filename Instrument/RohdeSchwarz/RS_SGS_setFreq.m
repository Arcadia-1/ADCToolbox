function RS_SGS_setFreq(rs, freq)

    RS_openConn(rs);

    if(freq < 10^6 || freq > 12.75*10^9)
        error('Rohde&Schwarz: SGS: Frequency out of range (1M ~ 12.75G)');
    end
    
    fprintf(rs, ['SOURce:FREQuency ',num2str(freq,'%.12f'),'Hz']);
    RS_waitComm;
    
    fclose(rs);
    
end
