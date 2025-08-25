function CH341_setIODir(dir,device_id)

% dir = [1:8], 0 for input, 1 for output

    if(nargin < 2)
        device_id = 0;
    end
    
    if(length(dir) ~= 8)
        error('CH341: Incorrect GPIO IO setting');
    end
    
    CH341_check(device_id);
    
    calllib('CH341DLLA64','CH341OpenDevice',device_id);
    
    iEnable = 8;
    iSetDirOut = sum((dir==1).*2.^(0:7));
    iSetDataOut = 0;
    
    calllib('CH341DLLA64','CH341SetOutput',device_id, iEnable, iSetDirOut, iSetDataOut);
    
    calllib('CH341DLLA64','CH341CloseDevice',device_id);
end