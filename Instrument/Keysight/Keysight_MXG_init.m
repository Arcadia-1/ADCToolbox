function [keysight]= Keysight_MXG_init(sn)

    keysight=visa('ni',sn);

    Keysight_reset(keysight);
   
    fopen(keysight);
    fprintf(keysight, ':OUTPut:MODulation OFF');
    fclose(keysight);

end





