function ok = Keysight_openConn(keysight)

    fclose(keysight);
    try
        fopen(keysight);
        ok = 1;
    catch E
        ok = 0;
        if(nargout == 0)
            error('Keysight: Cannot connect to device: %s',keysight.SerialNumber);
        end
    end

end