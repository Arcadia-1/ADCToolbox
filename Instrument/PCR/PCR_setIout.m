function ok = PCR_setIout(h, iout)

% iout is in uA
% iout <0 for current sink, >0 for current source
    
    if(iout < -256 || iout > 252)
        error('PCR: Iout setting out of range (-256uA~252uA)');
    end
    
    PCR_checkConn(h);
    
    if(iout >=0 && iout < 16)
        R = 0;
        IOUT = floor(iout/16*64);
        iout_acc = IOUT/64*16;
    elseif(iout <0 && iout >= -16)
        R = 0;
        IOUT = 128-floor(-iout/16*64);
        iout_acc = -(128-IOUT)/64*16;
    elseif(iout >= 16 && iout < 64)
        R = 1;
        IOUT = floor(iout/64*64);
        iout_acc = IOUT/64*64;
    elseif(iout < -16 && iout >= -64)
        R = 1;
        IOUT = 128-floor(-iout/64*64);
        iout_acc = -(128-IOUT)/64*64;
    elseif(iout > 0)
        R = 2;
        IOUT = floor(iout/256*64);
        iout_acc = IOUT/64*256;
    else
        R = 2;
        IOUT = 128-floor(-iout/256*64);
        iout_acc = -(128-IOUT)/64*256;
    end
    
    fprintf('PCR: Setting PCR@%s Iout to %.2fuA (range %d) ... ', h.addr_disp, iout_acc, R);
    
    CH341_I2C(h.addr, [hex2dec('E8'),0], 0);            % Iout set to 0
    CH341_I2C(h.addr, [hex2dec('E4'),R*64+1], 0);       % auto range
    CH341_I2C(h.addr, [hex2dec('E8'),IOUT], 0);         % Iout set to aim
    IOUT_RB = CH341_I2C(h.addr, [hex2dec('E8')], 1);
    
    if(IOUT_RB == IOUT)
        fprintf('Success!\n');
        ok = 1;
    else
        fprintf('Read back checking failed!\n');
        ok = 0;
    end
    
end