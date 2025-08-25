CH341_init();
clc;
[h1,h2] = PVR_init('00');
fprintf('Vsupply = %.3fV, %.3fV\n\n',PVR_getVsup(h1),PVR_getVsup(h2));
fprintf('Temperature = %.1fC, %.1fC\n\n',PVR_getTemp(h1),PVR_getTemp(h2));
PVR_setVout(h1,0.9); PVR_setVout(h2,1);
fprintf('Vout (off) = %.3fV, %.3fV\n',PVR_getVout(h1),PVR_getVout(h2));
fprintf('Iout (off) = %.0fuA, %.0fuA\n\n',PVR_getIout(h1)*10^6,PVR_getIout(h2)*10^6);
PVR_turn(h1,1); PVR_turn(h2,1);
pause(1);
fprintf('Vout (on) = %.3fV, %.3fV\n',PVR_getVout(h1),PVR_getVout(h2));
fprintf('Iout (on) = %.0fuA, %.0fuA\n\n',PVR_getIout(h1)*10^6,PVR_getIout(h2)*10^6);

PVR_checkError(h1); PVR_checkError(h2);

PVR_clearError(h1); PVR_clearError(h2);

pause(1);
PVR_turn(h1,0); PVR_turn(h2,0);