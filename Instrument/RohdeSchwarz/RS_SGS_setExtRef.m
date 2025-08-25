function RS_SGS_useExtRef(rs, ext)
    
    % ext = 1 : use external clk reference, = 0 : use internal OCXO

    RS_openConn(rs);

    if(ext)
        rs_send_command(rs, ':ROSCillator:SOURce EXTernal');
    else
        rs_send_command(rs, ':ROSCillator:SOURce INTernal');
    end

    RS_waitComm;
    
    fclose(rs);
    
end
