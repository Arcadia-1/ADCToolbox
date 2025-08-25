% this script is only for one-time EEPROM setup after PVR4 manufactory
% !!! don't run this on other board except PVR4!!!

addr = '11';    % addr = zz

CH341_init();
CH341_confI2C(1);
abs_addr = (sum((addr-'0').*[3,1]) + hex2dec('5C'))*2;

CH341_I2C(abs_addr, [hex2dec('16')], 0);

pause(1);

if(CH341_I2C(abs_addr, [hex2dec('98')], 1) ~= hex2dec('11'))
    error(['Cannot find a new PVRE@',addr]);
end

CH341_I2C(abs_addr, [hex2dec('E6'),hex2dec('75')],0);

abs_addr = (sum((addr-'0').*[3,1]) + hex2dec('75'))*2;

if(CH341_I2C(abs_addr, [hex2dec('98')], 1) ~= hex2dec('11'))
    error(['Lost connection to PVRE@',addr,' after changing abs addr']);
end

CH341_I2C(abs_addr, [hex2dec('15')], 0);

pause(1);

CH341_I2C(abs_addr, [hex2dec('16')], 0);

pause(1);

if(CH341_I2C(abs_addr, [hex2dec('98')], 1) ~= hex2dec('11'))
    error(['PVRE@',addr,' setup failed, please retry']);
else
    fprintf(['PVRE@',addr,' setup successfully!\n\n']);
end

