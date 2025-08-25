function ok = TL413_check()

    if (~libisloaded('TLSDK64'))
        error('CH341: Please call TL413_init before operating other TK413 functions');
    end

    err = calllib('TLSDK64', 'ulaSDKGetLastError');
    
    if(err == 0)
        ok = 1;
    elseif(err == 55)
        unloadlibrary('TLSDK64');
        error('TL413: Dependent DLLs not found. Please copy AqDataMng64.dll, AqLARunW64.dll and TLFwPack.bin to your run directory');
    else
        ok = 0;
        if(nargout == 0)
            error('TL413: An error 0x%s occured "%s"',dec2hex(err),TL413_errCode(err));
        end
    end

end