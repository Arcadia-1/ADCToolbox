CH341_init();
clc;
[h1,h2] = PVRE_init('00');
pause(1);
fprintf('Vsupply = %.3fV, %.3fV\n\n',PVRE_getVsup(h1),PVRE_getVsup(h2));
fprintf('Temperature = %.1fC, %.1fC\n\n',PVRE_getTemp(h1),PVRE_getTemp(h2));
fprintf('Vout (off) = %.3fV, %.3fV\n',PVRE_getVout(h1),PVRE_getVout(h2));
fprintf('Iout (off) = %.0fuA, %.0fuA\n\n',PVRE_getIout(h1)*10^6,PVRE_getIout(h2)*10^6);
PVRE_setVout(h1,1); PVRE_setVout(h2,3);
pause(1);
fprintf('Vout (on) = %.3fV, %.3fV\n',PVRE_getVout(h1),PVRE_getVout(h2));
fprintf('Iout (on) = %.0fuA, %.0fuA\n\n',PVRE_getIout(h1)*10^6,PVRE_getIout(h2)*10^6);

PVRE_checkError(h1); PVRE_checkError(h2);

PVRE_clearError(h1); PVRE_clearError(h2);

pause(1);
PVRE_setVout(h1,0); PVRE_setVout(h2,0);