function [temp] = PVR_getTemp(h)
    
    PVR_checkConn(h);
    
    TEMP = CH341_I2C(h.addr, [hex2dec('8D')], 2);
    temp = PVR_L5112dec(TEMP(2), TEMP(1));
    
end