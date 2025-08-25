function CH341_setGPIO(data,device_id)

% data = [1:8]

    if(nargin < 2)
        device_id = 0;
    end
    
    if(length(data) ~= 8)
        error('CH341: Incorrect GPIO data setting');
    end
    
    CH341_check(device_id);
    
    calllib('CH341DLLA64','CH341OpenDevice',device_id);
    
    iEnable = 4;
    iSetDirOut = 0;
    iSetDataOut = sum((data==1).*2.^(0:7));
    
    calllib('CH341DLLA64','CH341SetOutput',device_id, iEnable, iSetDirOut, iSetDataOut);
    
    calllib('CH341DLLA64','CH341CloseDevice',device_id);
end