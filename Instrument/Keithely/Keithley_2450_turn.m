function Keithley_2450_turn(kei, on)

Keithley_openConn(kei);

if(on)
    fprintf(kei,':OUTP ON');
else
    fprintf(kei,':OUTP OFF');
end

Keithley_waitComm;
fclose(kei);

end