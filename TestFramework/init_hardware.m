hw = struct();  % hardware handles

% !!write this part according to your actual setup!!
% be careful to the ID and address of the instruments used

hw.rigol_ps = RIGOL_init(RIGOL_PS_6);           % power supply
RIGOL_PS_setChannel(hw.rigol_ps, 1, 5, 0.2);    % set ch1 voltage and compliance
RIGOL_PS_setChannel(hw.rigol_ps, 2, 0.5, 0.2);  % set ch2 voltage and compliance
RIGOL_PS_turn(hw.rigol_ps, 1, 1, 0);            % turn on power supply

pause(1);

CH341_init;     % initialize the USB controller (CH341) on PPRS

[hw.pvr_Vref1, hw.pvr_Vana] = PVR_init('00');   % define reference cards - pay attention to the address and channel number
[hw.pvr_Vref2, hw.pvr_Vamp] = PVR_init('0z');
[hw.pvr_Vdig2, hw.pvr_Vdig1] = PVR_init('01');
[hw.pvr_Viol, hw.pvr_Vdig3] = PVR_init('z0');
[hw.pvr_Vclk, hw.pvr_Vioh] = PVR_init('zz');
[hw.pvr_Temp, ~] = PVR_init('11');
[hw.pcr_Iref1] = PCR_init('0z');
[hw.pcr_Iref2] = PCR_init('00');

hw.rigol_ss = RIGOL_init(RIGOL_SS_5);           % RIGOL AWG source (DG4000 series)
RIGOL_SS_setClkSrc(hw.rigol_ss, 1);             % using external clock source
RIGOL_SS_setImpedance(hw.rigol_ss, 50, 50);     % set output impedance to 50ohm

TL413_init();           % logic analyzer
TL413_setVTh(1.6);      % set threshold voltage 
TL413_setClkEdge(1);    % which clk edge to capture data, 0 for rising edge, 1 for falling edge

hw.apx = APX_555B_init();   % APX audio source

hw.keysight_psg = Keysight_PSG_init('101.6.65.6');  % Keysight RF source

pause(1);
