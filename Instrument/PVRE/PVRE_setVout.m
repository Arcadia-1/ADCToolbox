function ok = PVRE_setVout(h, vout)

    vtol = 0.01;
    
    if(vout < -2.8 || vout > 3)
        error('PVRE: Vout setting out of range (-2.8V~3V)');
    end
    
    vdd = PVRE_getVsup(h);
    
    vout = vdd/2 + vout*0.2*(1+24.2/100);

    PVRE_checkConn(h);
    
    [VOUT_MSB, VOUT_LSB] = PVR_dec2L16(vout);
    [VHI_MSB, VHI_LSB] = PVR_dec2L16(vout+vtol,+1);
    [VLOW_MSB, VLOW_LSB] = PVR_dec2L16(vout-vtol,-1);
    
    vout_acc = (PVR_L162dec(VOUT_MSB,VOUT_LSB)-vdd/2)/0.2/(1+24.2/100);
    vout_hi = (PVR_L162dec(VHI_MSB,VHI_LSB)-vdd/2)/0.2/(1+24.2/100);
    
    fprintf('PVR: Setting PVR@%s (ch #%d) Vout to %.3fV (+-%.3fV) ... ', h.addr_disp, h.ch, vout_acc, vout_hi-vout_acc);
    
    CH341_I2C(h.addr, [hex2dec('00'),h.ch], 0);
    
    CH341_I2C(h.addr, [hex2dec('21'), VOUT_LSB, VOUT_MSB], 0);
    CH341_I2C(h.addr, [hex2dec('25'), VOUT_LSB, VOUT_MSB], 0);
    CH341_I2C(h.addr, [hex2dec('26'), VOUT_LSB, VOUT_MSB], 0);
    CH341_I2C(h.addr, [hex2dec('42'), VHI_LSB, VHI_MSB], 0);
    CH341_I2C(h.addr, [hex2dec('43'), VLOW_LSB, VLOW_MSB], 0);
    
    CH341_I2C(h.addr, [hex2dec('01'),hex2dec('80')], 0);

    VOUT_RB = CH341_I2C(h.addr, [hex2dec('21')], 2);
    if(VOUT_RB(1) == VOUT_LSB && VOUT_RB(2) == VOUT_MSB)
        fprintf('Success!\n');
        ok = 1;
    else
        fprintf('Read back checking failed!\n');
        ok = 0;
    end
end