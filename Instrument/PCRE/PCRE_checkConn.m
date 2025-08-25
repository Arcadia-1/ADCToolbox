function ok = PCRE_checkConn(h)

    if(~strcmp(h.type,'PCRE'))
        error(['PCRE: Wrong handle type']);
    end
    
    if(CH341_I2C(h.addr, [hex2dec('98')], 1) == hex2dec('11'))
        ok = 1;
    else
        ok = 0;
        if(nargout == 0)
            error(['PCRE: Connection to PCRE@',h.addr_disp,' is broken']);
        end
    end

end