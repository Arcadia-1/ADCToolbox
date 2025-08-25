function ok = PCR_turn(h, on)

    PCR_checkConn(h);
    
    if(on)
        fprintf('PCR: Turning on PCR@%s ... ',h.addr_disp);
        CH341_I2C(h.addr, [hex2dec('01'),hex2dec('80')], 0);
        CH341_I2C(h.addr, [hex2dec('E2'),hex2dec('00')], 0);
        ST = CH341_I2C(h.addr, [hex2dec('01')], 1);
        if(ST == hex2dec('80'))
           fprintf('Success!\n');
           ok = 1;
        else
           fprintf('Read back checking failed!\n');
           ok = 0;
        end
    else
        fprintf('PCR: Turning off PCR@%s ... ',h.addr_disp);
        CH341_I2C(h.addr, [hex2dec('01'),hex2dec('00')], 0);
        CH341_I2C(h.addr, [hex2dec('E2'),hex2dec('01')], 0);
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