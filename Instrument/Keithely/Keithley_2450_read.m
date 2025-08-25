function [vout,iout] = Keithley_2450_read(kei)

    Keithley_openConn(kei);

    fprintf(kei,':FUNC "VOLTage"');
    Keithley_waitComm;
    fprintf(kei,':READ?');
    vout = fscanf(kei,'%g');
    
    fprintf(kei,':FUNC "CURRent"');
    Keithley_waitComm;
    fprintf(kei,':READ?');
    iout = fscanf(kei,'%g');
    
    fclose(kei);
end