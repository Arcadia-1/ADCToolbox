function CH341_confI2C(speed, device_id)

% speed: 0 for 20KHz, 1 for 100KHz, 2 for 400KHz, 3 for 750KHz
    
    if(nargin < 2)
        device_id = 0;
    end
    
    if(speed < 0 || speed > 3)
        error('CH341: I2C speed setting out of range');
    end
    
    CH341_check(device_id);
    
    calllib('CH341DLLA64','CH341OpenDevice',device_id);
    calllib('CH341DLLA64','CH341SetStream',device_id,speed);
    calllib('CH341DLLA64','CH341CloseDevice',device_id);
end