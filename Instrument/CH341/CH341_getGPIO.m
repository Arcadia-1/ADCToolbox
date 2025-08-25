function [dat, oth] = CH341_getGPIO(device_id)
    
    if(nargin < 1)
        device_id = 0;
    end
    
    CH341_check(device_id);
    
    calllib('CH341DLLA64','CH341OpenDevice',device_id);
    
    iStatus = libpointer('uint32Ptr', 0);
    calllib('CH341DLLA64','CH341GetInput',device_id,iStatus);
    
    bin = dec2bin(iStatus.value,24);
    bin = bin - '0';
    
    dat = bin(24:-1:17);
    oth = bin(16:-1:9);
    
    calllib('CH341DLLA64','CH341CloseDevice',device_id);
    clear iStatus
end
    