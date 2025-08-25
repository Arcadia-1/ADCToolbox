function TL413_newTrig()
    
    global TL413_iTrig
    
    TL413_check();

    TL413_iTrig = libstruct('SDKTRIG');
    TL413_iTrig.lpiCont = zeros(16,1);
    TL413_iTrig.lpbTrigData = zeros(2048,1);
    iTrigPtr = libpointer('SDKTRIG', TL413_iTrig);
    
    calllib('TLSDK64', 'ulaSDKClearTrigger', iTrigPtr);
    
    TL413_iTrig = iTrigPtr.value;
    clear iTrigPtr
    
    TL413_check();

end