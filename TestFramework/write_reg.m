function write_reg(pr, verify)

    % <parse your configurations to the actual bistreams (for scan chain)
    % or other protocal (SPI / I2C ...)
    reg_wr = [bitget(pr.stGnd,6),bitget(pr.stGnd,5),bitget(pr.stGnd,4),...
              bitget(pr.stGnd,3),bitget(pr.stGnd,2),bitget(pr.stGnd,1),...
              pr.refCap,pr.clkInBuf,pr.enOsc,pr.clkSel,pr.clkOut,...
              pr.outMode,pr.refStb,bitget(pr.prbSel,[3,2,1]),...
              bitget(pr.osrSel,[3,2,1]),bitget(pr.chopSel,[3,2,1]),...
              bitget(pr.timing,[5,6]), bitget(pr.timing,[3,4]),...
              bitget(pr.timing,[1,2]), bitget(pr.cycGap,[2,1]),...
              pr.enDel,pr.enRot];
          
    % RBIT, RCLK, RSS, WBIT, WCLK, WSS

    % set IO direction
    CH341_setIODir([0,1,1,1,1,1,1,1]);

    % write reg
    CH341_setGPIO([0,0,0,0,0,0,0,0]);
    pause(0.1);
    
    % write configurations
    CH341_openDevice;
    CH341_setGPIOFast([0,0,0,0,0,1,0,0]);   % setGPIOFast is faster than setGPIO
    for i1 = 1:32
        CH341_setGPIOFast([0,0,0,reg_wr(i1),0,1,0,0]);
        CH341_setGPIOFast([0,0,0,reg_wr(i1),1,1,0,0]);
    end
    CH341_setGPIOFast([0,0,0,0,0,1,0,0]);
    CH341_setGPIOFast([0,0,0,0,0,0,0,0]);
    CH341_closeDevice;
    
    % read back verification
    ok = verify_reg(pr,verify);
    
end