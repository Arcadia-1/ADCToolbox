function [keysight]= Keysight_WG_init(sn)

    keysight=visa('keysight',sn);

    Keysight_reset(keysight);

end





