function i_meas = PVRE_getIout(h)
    
    PVRE_checkConn(h);
    
    CH341_I2C(h.addr, [hex2dec('00'),h.ch], 0);
    
    IVOUT = CH341_I2C(h.addr, [hex2dec('FA')], 2);
    current_voltage_out = PVR_L162dec(IVOUT(2), IVOUT(1));
    i_meas = current_voltage_out*0.025/10;                  %0.75 if bit-8 at D0 is 1, R=10ohm
    
end