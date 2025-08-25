CH341_init();
clc;
[h1] = PCR_init('zz'); [h2] = PCR_init('11');
PCR_setIout(h1, -10);   % sink 10 uA
PCR_setIout(h2, 20);    % source 20 uA
PCR_turn(h1,0); PCR_turn(h2,0);  