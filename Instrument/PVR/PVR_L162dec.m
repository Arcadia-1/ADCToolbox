function [num] = PVR_L162dec(MSB, LSB)
    
    num = floor(double(LSB)+double(MSB)*256)/2^13;
    
end