function [ok] = CH341_check(device_id)
    
    if(nargin < 1)
        device_id = 0;
    end
    
    if (~libisloaded('CH341DLLA64'))
        error('CH341: Please call CH341_init before operating other CH341 functions');
    end
    calllib('CH341DLLA64','CH341OpenDevice',device_id);
    ver = calllib('CH341DLLA64','CH341GetVerIC',device_id);
    
    if(ver == 0)
        ok = 0;
        if(nargout == 0)
            error('CH341: Cannot connect to device #%d',device_id);
        end
    else
        ok = 1;
    end
    
    calllib('CH341DLLA64','CH341CloseDevice',device_id);

end