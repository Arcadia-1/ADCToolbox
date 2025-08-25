function TL413_stop()

    TL413_check;
    
    if(~calllib('TLSDK64', 'ulaSDKIsCaptureReady'))
        fprintf('TL413: Stop device ... ');
        calllib('TLSDK64', 'ulaSDKStopCapture');
        TL413_check;
        fprintf('OK!\n');
    end
    
end