function [i_meas] = PCRE_getIsup(h)
    
    PCRE_checkConn(h);
    
    ISUP = CH341_I2C(h.addr, [hex2dec('89')], 2);
    i_meas = PVR_L5112dec(ISUP(2), ISUP(1));
    
end