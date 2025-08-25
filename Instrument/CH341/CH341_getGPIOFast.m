function [dat, oth] = CH341_getGPIOFast(device_id)
    
    if(nargin < 1)
        device_id = 0;
    end
    
    
    iStatus = libpointer('uint32Ptr', 0);
    calllib('CH341DLLA64','CH341GetInput',device_id,iStatus);
    
    dec2hex(iStatus.value);
    bin = dec2bin(iStatus.value,24);
    bin = bin - '0';
    
    dat = bin(24:-1:17);
    oth = bin(16:-1:9);
    
    clear iStatus
end
    