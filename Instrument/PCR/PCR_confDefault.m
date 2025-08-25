function ok = PCR_confDefault(h)

    PCR_checkConn(h);
    
    fprintf('PCR: Writing default configurations to PCR@%s ... ', h.addr_disp);
    
    CH341_I2C(h.addr, [hex2dec('01'),hex2dec('00')], 0);    % turn off;
    CH341_I2C(h.addr, [hex2dec('E8'),hex2dec('00')], 0);    % output hi-z
    CH341_I2C(h.addr, [hex2dec('E2'),hex2dec('01')], 0);    % turn off LED
    CH341_I2C(h.addr, [hex2dec('E4'),hex2dec('01')], 0);    % low range, max slew
    CH341_I2C(h.addr, [hex2dec('E6'),hex2dec('40')], 0);    % no outupt limit
    
    fprintf('Done!\n');
    ok = 1;

end