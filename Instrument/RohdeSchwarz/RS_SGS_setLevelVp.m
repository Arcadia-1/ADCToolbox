function RS_SGS_setLevelVp(rs, lvl)

    % level in Vp

    RS_openConn(rs);

    if(lvl < 0 || lvl > 5.6)
        error('Rohde&Schwarz: SGS: Level out of range (0 ~ 5.6V)');
    end
    
    fprintf(rs, ['POWer ',num2str(lvl,'%.6f'),'V']);
    RS_waitComm;
    
    fclose(rs);
    
end
