function data = TL413_getDataBus(MSB,LSB)

% capture data of channels [MSB:LSB]
% output data is n-row by 1 ([MSB:LSB] are combined as decimal)

    if(max(MSB) > 31 || min(MSB) < 0)
        error('TL413: MSB out of range (0~31)');
    end
    if(max(LSB) > 31 || min(LSB) < 0)
        error('TL413: LSB out of range (0~31)');
    end
    if(MSB < LSB)
        error('TL413: MSB cannot be smaller than LSB');
    end
    
    TL413_wait;
    
    samNum = calllib('TLSDK64', 'ulaSDKGetSamplesNum');
    TL413_check;
    
    fprintf('TL413: %.1f kbit data avaliable.\n  Allocating memory ... ',(MSB-LSB+1)*samNum/1000);
    iBufPtr = libpointer('uint32Ptr', zeros(samNum, 1));
    iBufSize = libpointer('int32Ptr', samNum);
    
    fprintf('\n  Receiving data ... ');
    calllib('TLSDK64', 'ulaSDKGetBusData', MSB, LSB, iBufPtr, iBufSize, 0);
    TL413_check;
    
    data = iBufPtr.value;

    fprintf('\n  Releasing memory ... ');
    clear iBufPtr
    clear iBufSize
    
    fprintf('\n  Done!\n');
    
end