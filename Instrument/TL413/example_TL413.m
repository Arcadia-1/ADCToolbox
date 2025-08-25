TL413_init;
TL413_setMode(0);
TL413_setVTh(0.5);
TL413_setSamRate(1*10^6);
TL413_setSamNum(1000);
TL413_setTrig(0,1,0);
TL413_run;
data = TL413_getData([0,1,2]);
plot(data);
axis([1,1000,-1,2])
% TL413_stop;
% TL413_close;

