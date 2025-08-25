function [kei]= Keithley_init(sn)

    kei=visa('ni',sn);

    Keithley_reset(kei);
    
    Keithley_openConn(kei);
    fprintf(kei,':OUTP:CURR:SMOD HIMP');
    Keithley_waitComm;
    fprintf(kei,':OUTP:VOLT:SMOD HIMP');
    Keithley_waitComm;
    fprintf(kei,':TRIG:CONT AUTO');
    Keithley_waitComm;
    
    fclose(kei);
end