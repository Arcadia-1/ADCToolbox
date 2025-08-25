CH341_init();
clc;
[h1,h2] = PCRE_init('00');
pause(1);
fprintf('Vsupply = %.3fV, %.3fV\n\n',PCRE_getVsup(h1),PCRE_getVsup(h2));
fprintf('Temperature = %.1fC, %.1fC\n\n',PCRE_getTemp(h1),PCRE_getTemp(h2));
fprintf('Vout (on) = %.3fV, %.3fV\n',PCRE_getVout(h1),PCRE_getVout(h2));
fprintf('Iout (on) = %.1fuA, %.1fuA\n\n',PCRE_getIout(h1)*10^6,PCRE_getIout(h2)*10^6);
PCRE_setIout(h1,-40); PCRE_setIout(h2,80);
pause(1);
fprintf('Vout (on) = %.3fV, %.3fV\n',PCRE_getVout(h1),PCRE_getVout(h2));
fprintf('Iout (on) = %.1fuA, %.1fuA\n\n',PCRE_getIout(h1)*10^6,PCRE_getIout(h2)*10^6);

PCRE_checkError(h1); PCRE_checkError(h2);

PCRE_clearError(h1); PCRE_clearError(h2);

pause(1);
PCRE_setIout(h1,0); PCRE_setIout(h2,0);