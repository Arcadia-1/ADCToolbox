function PVRE_clearError(h)

    PVRE_checkConn(h);
    
    CH341_I2C(h.addr, [hex2dec('03')], 0);

end