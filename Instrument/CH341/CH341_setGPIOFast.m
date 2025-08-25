function CH341_setGPIOFast(data,device_id)

% data = [1:8]

    if(nargin < 2)
        device_id = 0;
    end
    
    if(length(data) ~= 8)
        error('CH341: Incorrect GPIO data setting');
    end
    
    iEnable = 4;
    iSetDirOut = 0;
    iSetDataOut = sum((data==1).*2.^(0:7));
    
    calllib('CH341DLLA64','CH341SetOutput',device_id, iEnable, iSetDirOut, iSetDataOut);

end