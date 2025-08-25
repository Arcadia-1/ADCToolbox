function ok = RIGOL_openConn(rigol)

    fclose(rigol);
    try
        fopen(rigol);
        ok = 1;
    catch E
        ok = 0;
        if(nargout == 0)
            error('RIGOL: Cannot connect to device: %s',rigol.SerialNumber);
        end
    end

end