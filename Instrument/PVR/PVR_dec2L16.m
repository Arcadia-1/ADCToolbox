function [MSB, LSB] = PVR_dec2L16(num, offset)

    if(nargin < 2)
        offset = 0;
    end
    if(num < 0 || num > 7.999) 
        error('dec2L16: num should be within [0,8)');
    end

    DIG = max(round(num*2^13) + offset,0);
    
    MSB = floor(DIG/256);
    LSB = mod(DIG,256);

end