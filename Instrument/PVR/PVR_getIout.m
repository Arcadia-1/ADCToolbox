function i_meas = PVR_getIout(h)
    
    PVR_checkConn(h);

    [MSB, LSB] = PVR_dec2L511(1000);       % R in mohm, 1ohm maximum
    
    CH341_I2C(h.addr, [hex2dec('00'),h.ch], 0);

    CH341_I2C(h.addr, [hex2dec('38'), LSB, MSB], 0);

    IOUT = CH341_I2C(h.addr, [hex2dec('8C')], 2);
    i_meas = PVR_L5112dec(IOUT(2), IOUT(1))/ 10; % actual R = 10ohm
    
%     IVOUT = CH341_I2C(h.addr, [hex2dec('FA')], 2);
%     current_voltage_out = PVR_L162dec(IVOUT(2), IVOUT(1));
%     i_meas = current_voltage_out*0.025/10;                  %0.75 if bit-8 at D0 is 1, R=10ohm
    
end