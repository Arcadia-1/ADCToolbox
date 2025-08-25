function [sn,idev] = TL413_init()

    if (~libisloaded('TLSDK64'))    
        fprintf('Loading TL413 library... ');
        loadlibrary('TLSDK64');
        fprintf('OK!\n');
    end
    
    calllib('TLSDK64', 'ulaSDKClose');
    
    fprintf('TL413: Initializing device ... ');
    calllib('TLSDK64', 'ulaSDKInit');
    
    TL413_check;
    
    fprintf('OK!\n');
    
    idev = calllib('TLSDK64', 'ulaSDKGetLaID');
    TL413_check;

    iSN = libpointer('uint8Ptr', zeros([1,50]));
    calllib('TLSDK64', 'ulaSDKGetSerialNumber', iSN, 50);
    sn = erase(char(iSN.value),char(0));
    clear iSN
    
    fprintf('  Device ID: %s\n',dec2hex(idev));
    fprintf('  Device SN: %s\n',sn);
    
    if(idev ~= hex2dec('41340'))
        fprintf('  Warning: This Matlab lib is developed for TL4134E. No gurantee on other models\n\n');
    end
    
    TL413_setMode(0);               % default mode: sychronous sampling
    TL413_setVTh(1.2);              % default vth: 1.2V
    TL413_setSamNum(0);             % default sample number: max 
    TL413_setSamRate(200*10^6);     % default sampling rate: 200MHz
    TL413_setClkEdge(0);            % default sampling edge: rising
    TL413_setTrigPass(0);           % default pass count: 0
    TL413_setTrigPos(1);            % default trig pos: 1
    
    TL413_newTrig;                  % init iTrig
    TL413_setTrig(0, 0, 0) % default trig: no condition
    
end