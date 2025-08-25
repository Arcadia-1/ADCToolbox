function ok = PCR_checkConn(h)

    if(~strcmp(h.type,'PCR'))
        error(['PCR: Wrong handle type']);
    end
    
    if(CH341_I2C(h.addr, [hex2dec('98')], 1) == hex2dec('22'))
        ok = 1;
    else
        ok = 0;
        if(nargout == 0)
            error(['PCR: Connection to PCR@',h.addr_disp,' is broken']);
        end
    end

end