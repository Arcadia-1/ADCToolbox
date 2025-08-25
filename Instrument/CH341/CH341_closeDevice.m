function [ok] = CH341_closeDevice(device_id)

    if(nargin < 1)
        device_id = 0;
    end
    
    ok = CH341_check(device_id);
    
    calllib('CH341DLLA64','CH341CloseDevice',device_id);
    
end