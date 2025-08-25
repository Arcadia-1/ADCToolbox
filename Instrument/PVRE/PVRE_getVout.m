function [v_meas, v_set, rel_err] = PVRE_getVout(h)
    
    PVRE_checkConn(h);
    
    CH341_I2C(h.addr, [hex2dec('00'),h.ch], 0);
    
    VOUT = CH341_I2C(h.addr, [hex2dec('8B')], 2);
    VOUT_RB = CH341_I2C(h.addr, [hex2dec('21')], 2);
    
    vdd = PVRE_getVsup(h);
    
    v_meas = (PVR_L162dec(VOUT(2), VOUT(1))-vdd/2)/0.2/(1+24.2/100);
    v_set = (PVR_L162dec(VOUT_RB(2), VOUT_RB(1))-vdd/2)/0.2/(1+24.2/100);
    rel_err = (v_meas-v_set)/v_set;
    
end