function ok = PCRE_setIout(h, iout)

    % iout in uA

    vtol = 0.01;
    
    if(iout < -50 || iout > 80)
        error('PCRE: Iout setting out of range (-50uA~80uA)');
    end

    PCRE_checkConn(h);
    
    vdd = PCRE_getVsup(h);
    vout = vdd/2 + iout*10^-6 * 510*(1+24.2/1);
    [VOUT_MSB, VOUT_LSB] = PVR_dec2L16(vout);
    [VHI_MSB, VHI_LSB] = PVR_dec2L16(vout+vtol,+1);
    [VLOW_MSB, VLOW_LSB] = PVR_dec2L16(vout-vtol,-1);
    
    vout_acc = PVR_L162dec(VOUT_MSB,VOUT_LSB);
    iout_acc = (vout_acc - vdd/2)/(510*(1+24.2/1))*10^6;
    
    fprintf('PCRE: Setting PCRE@%s (ch #%d) Iout to %.2fuA  ... ', h.addr_disp, h.ch, iout_acc);
    
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