function ok = PVR_confDefault(h)

    PVR_checkConn(h);
    
    fprintf('PVR: Writing default configurations to PVR@%s (ch #%d) ... ', h.addr_disp, h.ch);
    
    CH341_I2C(h.addr, [hex2dec('00'),h.ch], 0);                         % select page
    CH341_I2C(h.addr, [hex2dec('02'),hex2dec('1B')], 0);                % no Control pin, no off delay
    CH341_I2C(h.addr, [hex2dec('35'),hex2dec('00'),hex2dec('CA')], 0);  % VIN_ON is 4V
    CH341_I2C(h.addr, [hex2dec('36'),hex2dec('00'),hex2dec('C3')], 0);  % VIN_OFF is 3V
    CH341_I2C(h.addr, [hex2dec('40'),hex2dec('48'),hex2dec('79')], 0);  % VOUT_MAX is 3.8V
    CH341_I2C(h.addr, [hex2dec('44'),hex2dec('00'),hex2dec('00')], 0);  % VOUT_IN is 0V
    CH341_I2C(h.addr, [hex2dec('5E'),hex2dec('01'),hex2dec('00')], 0);  % PG_ON is 0V
    CH341_I2C(h.addr, [hex2dec('5F'),hex2dec('01'),hex2dec('00')], 0);  % PG_OFF is 0V
    CH341_I2C(h.addr, [hex2dec('62'),hex2dec('00'),hex2dec('80')], 0);  % no power on time checking
    CH341_I2C(h.addr, [hex2dec('CB'),hex2dec('43'),hex2dec('00')], 0);  % PG pin indicates output state   
    CH341_I2C(h.addr, [hex2dec('D0'),hex2dec('43'),hex2dec('00')], 0);  
    % DAC positive servo, max vout = 2.65, no continuous servo, current
    % sensing IR < +-170mV, max vout measure = 3.8V,  fast servo
    CH341_I2C(h.addr, [hex2dec('E8'),hex2dec('E8'),hex2dec('03')], 0);  % Isup measurement R = 1ohm
    CH341_I2C(h.addr, [hex2dec('38'),hex2dec('E8'),hex2dec('03')], 0);  % Iout measurement R = 1ohm
    
    fprintf('Done!\n');
    ok = 1;

end