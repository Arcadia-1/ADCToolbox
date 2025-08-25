function Keithley_2450_setIout(kei, iout, vlim)

    Keithley_openConn(kei);
    
    if(iout < -10E-3 || iout > 10E-3)      % limit to 10mA for protection, hardware support 1A max
        error('Keithely-2450: Current setting out of range (+-10mA)');
    end
    if(vlim < 0 || vlim > 2)                % limit to 2V for protection, hardware support 210V max
        error('Keithely-2450: Voltage limit setting out of range (0~2V)');
    end
    
    fprintf(kei,':SOUR:FUNC CURR');
    fprintf(kei,':SOUR:CURR:RANG:AUTO ON');
    fprintf(kei,sprintf(':SOUR:CURR:VLIM %g',vlim));
    fprintf(kei,sprintf(':SOUR:CURR %g',iout));
    Keithley_waitComm;

    fclose(kei);

end