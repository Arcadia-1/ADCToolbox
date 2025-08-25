rigol_ss = RIGOL_init(RIGOL_SS_5);  % initialize handle to rigol signal source #5

RIGOL_SS_setClkSrc(rigol_ss, 0);   % using internal clock source

RIGOL_SS_setImpedance(rigol_ss, 50, 0);  % set output impedance of channel 1 to 50ohm, channel 2 to lowest (high-z mode)

RIGOL_SS_setSine(rigol_ss,1,1*10^6,1,0,0); % set channel 1 to output sinewave, freq = 1M, Vp = 1V, dc offset = 0, phase = 0deg

RIGOL_SS_setPulse(rigol_ss, 2, 3.3, 0, 20*10^-6, 1*10^-6, 5*10^-6, 0.1*10^6, 0.1*10^-6);  
% set channel 2 to output pulse, period = 20u, vhigh = 3.3, vlow = 0, pulse width = 5u, rising and falling time = 0.1u

RIGOL_SS_turn(rigol_ss,1,1);   % turn on channel 1 and 2

RIGOL_SS_align(rigol_ss);  % align the phase of two channel