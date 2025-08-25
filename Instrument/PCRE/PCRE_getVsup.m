function [v_meas] = PCRE_getVsup(h)
    
    PCRE_checkConn(h);
    
    VSUP = CH341_I2C(h.addr, [hex2dec('88')], 2);
    v_meas = PVR_L5112dec(VSUP(2), VSUP(1));
    
end