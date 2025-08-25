function ok = PVR_setIoutMeasRange(h,range)

% range = 0, max measured Iout = 17mA; 
%         1, max measured Iout = 600mA;

    PVR_checkConn(h);
    
    CH341_I2C(h.addr, [hex2dec('00'),h.ch], 0);
    
    RG = CH341_I2C(h.addr, [hex2dec('D0')], 2);   
    
    if(range)
    
        fprintf('PVR: Setting PVR@%s (ch #%d) Iout measurement range to 600mA ... ', h.addr_disp, h.ch);
        
        RG(2) = bitor(RG(2),1);
    
    else
        
        fprintf('PVR: Setting PVR@%s (ch #%d) Iout measurement range to 17mA ... ', h.addr_disp, h.ch);
    
        RG(2) = bitand(RG(2),bin2dec('1111 1110'));
    
    end
    
    CH341_I2C(h.addr, [hex2dec('D0'),RG(1),RG(2)], 0);
    RB = CH341_I2C(h.addr, [hex2dec('D0')], 2);    
    if(RB(1) == RG(1) && RB(2) == RG(2))
        fprintf('Success!\n');
        ok = 1;
    else
        fprintf('Read back checking failed!\n');
        ok = 0;
    end

end