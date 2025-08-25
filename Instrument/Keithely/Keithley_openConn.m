function ok = Keithley_openConn(kei)

    fclose(kei);
    try
        fopen(kei);
        ok = 1;
    catch E
        ok = 0;
        if(nargout == 0)
            error('Keithley: Cannot connect to device: %s',kei.SerialNumber);
        end
    end

end