function [num] = PVR_L5112dec(MSB, LSB)

    bin = dec2bin(double(LSB)+double(MSB)*256,16);
    suf = bin2dec(bin(1:5))-(bin(1) == '1')*2^5;
    dig = bin2dec(bin(6:16))-(bin(6) == '1')*2^11;
    num = dig*2^suf;

end