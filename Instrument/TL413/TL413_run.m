function TL413_run()

    global TL413_iTrig

    TL413_check;
    
    if(~calllib('TLSDK64', 'ulaSDKIsCaptureReady'))
       fprintf('TL413: Device is running already.\n'); 
    else
       fprintf('TL413: Device start capturing ... ');
       TL413_refreshTrig;
       iTrigPtr = libpointer('SDKTRIG', TL413_iTrig);
       calllib('TLSDK64', 'ulaSDKCapture', iTrigPtr);
       clear iTrigPtr
       TL413_check;
       fprintf('OK!\n');
    end
    
end