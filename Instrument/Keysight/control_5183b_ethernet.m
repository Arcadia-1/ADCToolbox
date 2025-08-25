%vt = visa('keysight','TCPIP0::101.6.64.209::inst0::INSTR')
%fopen (vt);
%connect(vt)

%input frequency
v_pp = 200e-3;
frequency_in = 123999.1231;
%power_in_dbm = 10*log10(v_pp^2/(8*50)*1000);
%connect to N5183B signal generator
deviceObject = tcpip('101.6.65.6',5025);
deviceObject.OutputBufferSize = 100000;
deviceObject.ByteOrder = 'littleEndian';
deviceObject.Timeout = 10.0;
fopen(deviceObject);
fprintf(deviceObject,':OUTPut:STATe OFF')
fprintf(deviceObject,':OUTPut:MODulation:STATe OFF')
%fprintf(deviceObject,':OUTPut:STATe ON')
%fprintf(deviceObject,':SOURce:RADio:ARB:STATe OFF')
fprintf(deviceObject, ['SOURce:FREQuency ',num2str(frequency_in)]);
fprintf(deviceObject, ['POWer ',num2str(0)]);
%{
fprintf(deviceObject, ['UNIT:POWer? '];
fscanf()
%}
%fprintf(deviceObject,':OUTPut:STATe ON')
a = 1
fclose(deviceObject);
delete(deviceObject); 
clear deviceObject