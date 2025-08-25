function RS_SGS_setLeveldBm(rs, lvl)

    % level in dBm

    RS_openConn(rs);

    if(lvl < -120 || lvl > 25)
        error('Rohde&Schwarz: SGS: Level out of range (-120dB ~ 25dB)');
    end
    
    fprintf(rs, ['POWer ',num2str(lvl,'%.2f'),'dBm']);
    RS_waitComm;
    
    fclose(rs);
    
end
