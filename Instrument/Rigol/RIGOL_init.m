function [rigol]= RIGOL_init(sn)

    rigol=visa('ni',sn);

    RIGOL_reset(rigol);

end





