function ok = PVR_checkConn(h)

    if(~strcmp(h.type,'PVR'))
        error(['PVR: Wrong handle type']);
    end
    
    if(CH341_I2C(h.addr, [hex2dec('98')], 1) == hex2dec('11'))
        ok = 1;
    else
        ok = 0;
        if(nargout == 0)
            error(['PVR: Connection to PVR@',h.addr_disp,' is broken']);
        end
    end

end