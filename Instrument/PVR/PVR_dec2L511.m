function [MSB, LSB] = PVR_dec2L511(num)

    if(num < -2^25 || num >= 2^25) 
        error('dec2L511: num should be within (-2^25, 2^25)');
    end

    if(num == 0)
        B = 0;
        Y = 0;
    else
        B = -min(floor(log2(2^10/abs(num))),15);
        Y = floor(num/2^B);
    end
    
    if(B < 0)
        B = B + 32;
    end
    if(Y < 0)
        Y = Y + 2048;
    end
    
    DIG = B * 2^11 + Y;
    MSB = floor(DIG/256);
    LSB = mod(DIG,256);

end