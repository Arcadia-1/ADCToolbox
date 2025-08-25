function [ok] = CH341_init(device_id)

    if(nargin < 1)
        device_id = 0;
    end
    
    if (~libisloaded('CH341DLLA64'))
        fprintf('Loading CH341 library... ');
        loadlibrary('CH341DLLA64');
        fprintf('OK!\n');
    end
    
    ok = CH341_check(device_id);
    if(ok)
        fprintf('CH341: Connect to device #%d successfully!\n',device_id);
    else
        error('CH341: Cannot connect to device #%d',device_id);
    end

    CH341_confI2C(0); % default I2C speed 20KHz
end