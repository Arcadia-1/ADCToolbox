function Keysight_MXG_turn(keysight, on_off)

    fopen(keysight);
    if(on_off)
        fprintf(keysight,':OUTPut:STATe ON');
    else
        fprintf(keysight,':OUTPut:STATe OFF');
    end
    fclose(keysight);


end