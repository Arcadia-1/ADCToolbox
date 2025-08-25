function v_meas = PCRE_getVout(h)
    
    PCRE_checkConn(h);
    
    CH341_I2C(h.addr, [hex2dec('00'),h.ch], 0);
    
    IVOUT = CH341_I2C(h.addr, [hex2dec('FA')], 2);
    current_voltage_out = PVR_L162dec(IVOUT(2), IVOUT(1));
    v_meas = current_voltage_out*0.75;                  %0.75 if bit-8 at D0 is 1, R=10ohm
   
    
end