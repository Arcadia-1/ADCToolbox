function [error] = PVR_checkError(h)

    PVR_checkConn(h);
    
    CH341_I2C(h.addr, [hex2dec('00'),h.ch], 0);
    
    STATE_ALL = CH341_I2C(h.addr, [hex2dec('79')], 2);
    STATE_VOUT = CH341_I2C(h.addr, [hex2dec('7A')], 1);
    STATE_IOUT = CH341_I2C(h.addr, [hex2dec('7B')], 1);
    STATE_INPUT = CH341_I2C(h.addr, [hex2dec('7C')], 1);
    STATE_TEMP = CH341_I2C(h.addr, [hex2dec('7D')], 1);
    STATE_CML = CH341_I2C(h.addr, [hex2dec('7E')], 1);
    STATE_MFR = CH341_I2C(h.addr, [hex2dec('80')], 1);
    
    if(nargout == 0)
        fprintf('PVR: Error State of PVR@%s (ch #%d) ------------------\n',h.addr_disp, h.ch);
        fprintf('Overall: %s %s\n', dec2bin(STATE_ALL(1),8), dec2bin(STATE_ALL(2),8));
        fprintf('Vout:    %s\n', dec2bin(STATE_VOUT,8));
        fprintf('Iout:    %s\n', dec2bin(STATE_IOUT,8));
        fprintf('Input:   %s\n', dec2bin(STATE_INPUT,8));
        fprintf('Temp:    %s\n', dec2bin(STATE_TEMP,8));
        fprintf('CML:     %s\n', dec2bin(STATE_CML,8));
        fprintf('MFR:     %s\n', dec2bin(STATE_MFR,8));
        fprintf('---------------------------------------------------\n');
    end
    
    error = STATE_ALL;
end