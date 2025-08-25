function ok = PVRE_checkConn(h)

    if(~strcmp(h.type,'PVRE'))
        error(['PVRE: Wrong handle type']);
    end
    
    if(CH341_I2C(h.addr, [hex2dec('98')], 1) == hex2dec('11'))
        ok = 1;
    else
        ok = 0;
        if(nargout == 0)
            error(['PVRE: Connection to PVRE@',h.addr_disp,' is broken']);
        end
    end

end