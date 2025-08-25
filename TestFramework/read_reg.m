function [reg_rd, data_rd] = read_reg()

    % RBIT, RCLK, RSS, WBIT, WCLK, WSS
    
    CH341_setIODir([0,1,1,1,1,1,1,1]);
    
    data_rd = zeros([1,32]);
    reg_rd = zeros([1,32]);
        
    CH341_setGPIO([0,0,0,0,0,0,0,0]);
    pause(0.1);
    
    CH341_openDevice;
    CH341_setGPIOFast([0,0,1,0,0,0,0,0]);
    for i1 = 1:32
        CH341_setGPIOFast([0,1,1,0,0,0,0,0]);
        CH341_setGPIOFast([0,0,1,0,0,0,0,0]);
        io = CH341_getGPIOFast();
        data_rd(i1) = io(1);
    end
    for i1 = 1:32
        CH341_setGPIOFast([0,1,1,0,0,0,0,0]);
        CH341_setGPIOFast([0,0,1,0,0,0,0,0]);
        io = CH341_getGPIOFast();
        reg_rd(i1) = io(1);
    end
    CH341_setGPIOFast([0,0,0,0,0,0,0,0]);
    CH341_closeDevice;

end