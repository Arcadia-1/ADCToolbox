function PCRE_reTrim(h)

    PCRE_checkConn(h);
    
    fprintf('PCRE: Re-triming PCRE@%s (ch #%d) ... ', h.addr_disp, h.ch);
    
    CH341_I2C(h.addr, [hex2dec('00'),h.ch], 0);
    
    RG = CH341_I2C(h.addr, [hex2dec('D0')], 2);   
    
    CH341_I2C(h.addr, [hex2dec('D0'), bitor(RG(1),bin2dec('1000 0000')), RG(2)], 0);
    
    pause(0.5);
    
    CH341_I2C(h.addr, [hex2dec('D0'), bitand(RG(1),bin2dec('0111 1111')), RG(2)], 0);
    
    fprintf('Done!\n');

end