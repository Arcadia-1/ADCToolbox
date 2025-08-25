function RS_SGS_turn(rs, on)

    RS_openConn(rs);

    if(on)
        rs_send_command(rs, ':OUTPut:STATe 1');
    else
        rs_send_command(rs, ':OUTPut:STATe 0');
    end

    RS_waitComm;
    
    fclose(rs);
    
end
