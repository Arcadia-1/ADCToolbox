function Keysight_PSG_set(keysight, freq, pwr)

    fclose(keysight);
    fopen(keysight);
    fprintf(keysight, ['SOURce:FREQuency ',num2str(freq)]);
    fprintf(keysight, ['POWer ',num2str(pwr)]);
    fclose(keysight);

end