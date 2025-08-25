function [ok] = CH341_openDevice(device_id)

    if(nargin < 1)
        device_id = 0;
    end
    
    ok = CH341_check(device_id);
    
    calllib('CH341DLLA64','CH341OpenDevice',device_id);
    
end