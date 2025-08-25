function TL413_setSamNum(num)

% set num = 0 to bet maximum 
    
    max_num = calllib('TLSDK64', 'ulaSDKGetMaxSamplesNum');

    TL413_check();

    if(num == 0)
        num = max_num;
    elseif(num < 0 || num > max_num)
        error('TL413: Number of sample setting out of range. Maximum avaliable is %d',max_num);
    end
    
    num = floor(num);
    fprintf('TL413: Setting number of sample to %d (MAX = %d) ... ', num, max_num);
    calllib('TLSDK64', 'ulaSDKSetSamplesNum', num);
    
    TL413_check();
    
    fprintf('OK!\n');

end