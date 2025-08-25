function ok = PVR_turn(h, on)
    
    PVR_checkConn(h);
    
    CH341_I2C(h.addr, [hex2dec('00'),h.ch], 0);
    
    if(on)
        fprintf('PVR: Turning on PVR@%s ... ',h.addr_disp);
        CH341_I2C(h.addr, [hex2dec('01'),hex2dec('80')], 0);
        ST = CH341_I2C(h.addr, [hex2dec('01')], 1);
        if(ST == hex2dec('80'))
           fprintf('Success!\n');
           ok = 1;
        else
           fprintf('Read back checking failed!\n');
           ok = 0;
        end
    else
        fprintf('PVR: Turning off PVR@%s ... ',h.addr_disp);
        CH341_I2C(h.addr, [hex2dec('01'),hex2dec('00')], 0);
        ST = CH341_I2C(h.addr, [hex2dec('01')], 1);
        if(ST == hex2dec('00'))
           fprintf('Success!\n');
           ok = 1;
        else
           fprintf('Read back checking failed!\n');
           ok = 0;
        end
    end
    
end