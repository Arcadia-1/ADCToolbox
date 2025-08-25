function data = TL413_getData(chs)

% chs (0~31) is the channel to be capture, chs can be a vector
% output data is n-row by m-col, m is channel number, n is sample number

    if(max(chs) > 31 || min(chs) < 0)
        error('TL413: Channel ID out of range (0~31)');
    end
    
    TL413_wait;
    
    samNum = calllib('TLSDK64', 'ulaSDKGetSamplesNum');
    TL413_check;
    
    chNum = length(chs);
    
    fprintf('TL413: %.1f kbit data avaliable.\n  Allocating memory ... ',chNum*samNum/1000);
    iBufPtr = libpointer('uint32Ptr', zeros(samNum, 1));
    iBufSize = libpointer('int32Ptr', samNum);
    data = zeros([samNum,chNum]);
    
    fprintf('\n  Receiving data ');
    for i1 = 1:chNum
        fprintf('.');
        calllib('TLSDK64', 'ulaSDKGetChData', chs(i1), iBufPtr, iBufSize, 0);
        TL413_check;
        data(1:samNum,i1) = iBufPtr.value;
    end

    fprintf('\n  Releasing memory ... ');
    clear iBufPtr
    clear iBufSize
    
    fprintf('\n  Done!\n');
    
end