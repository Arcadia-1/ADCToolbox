function [v_meas, v_set, rel_err] = PVR_getVout(h)
    
    PVR_checkConn(h);
    
    CH341_I2C(h.addr, [hex2dec('00'),h.ch], 0);
    
    VOUT = CH341_I2C(h.addr, [hex2dec('8B')], 2);
    VOUT_RB = CH341_I2C(h.addr, [hex2dec('21')], 2);
    
    v_meas = PVR_L162dec(VOUT(2), VOUT(1));
    v_set = PVR_L162dec(VOUT_RB(2), VOUT_RB(1));
    rel_err = (v_meas-v_set)/v_set;
    
end