function [i_meas, i_set, rel_err] = PCRE_getIout(h)
    
    PCRE_checkConn(h);
    
    CH341_I2C(h.addr, [hex2dec('00'),h.ch], 0);
    
    vdd = PCRE_getVsup(h);
    
    VOUT = CH341_I2C(h.addr, [hex2dec('8B')], 2);
    VOUT_RB = CH341_I2C(h.addr, [hex2dec('21')], 2);
    
    v_meas = PVR_L162dec(VOUT(2), VOUT(1));
    i_meas = (v_meas - vdd/2)/(510*(1+24.2/1));
    v_set = PVR_L162dec(VOUT_RB(2), VOUT_RB(1));
    i_set = (v_set - vdd/2)/(510*(1+24.2/1));
    rel_err = (i_meas-i_set)/i_set;
    
end