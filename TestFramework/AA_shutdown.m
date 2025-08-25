% shutdown all devices' output, useful when leave the lab or switch chip

if(exist('hw','var'))
    CH341_setIODir([0,0,0,0,0,0,0,0]);
    CH341_setGPIO([0,0,0,0,0,0,0,0]);
    
    RIGOL_PS_turn(hw.rigol_ps, 0, 0, 0);
    RIGOL_SS_turn(hw.rigol_ss, 0, 0);
    
    APX_turn(hw.apx,0);
    
    TL413_stop;
    TL413_close;
    
    Keysight_PSG_turn(hw.keysight_psg,0);
    delete(hw.keysight_psg);
    
    clear hw
end