function ok = RS_openConn(rs)

    fclose(rs);
    try
        fopen(rs);
        ok = 1;
    catch E
        ok = 0;
        if(nargout == 0)
            error('Rohde&Schwarz: Cannot connect to device: %s',rs.SerialNumber);
        end
    end

end