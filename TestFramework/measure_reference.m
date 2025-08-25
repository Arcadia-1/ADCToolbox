% measure all voltages and currents
da.IVana = PVR_getIout(hw.pvr_Vana);
da.IVamp = PVR_getIout(hw.pvr_Vamp);
da.IVref1 = PVR_getIout(hw.pvr_Vref1);
da.IVref2 = PVR_getIout(hw.pvr_Vref2);
da.IVdig1 = PVR_getIout(hw.pvr_Vdig1);
da.IVdig2 = PVR_getIout(hw.pvr_Vdig2);
da.IVdig3 = PVR_getIout(hw.pvr_Vdig3);
da.IViol = PVR_getIout(hw.pvr_Viol);
da.IVioh = PVR_getIout(hw.pvr_Vioh);
da.IVclk = PVR_getIout(hw.pvr_Vclk);
[da.VVana] = PVR_getVout(hw.pvr_Vana);
[da.VVamp] = PVR_getVout(hw.pvr_Vamp);
[da.VVref1] = PVR_getVout(hw.pvr_Vref1);
[da.VVref2] = PVR_getVout(hw.pvr_Vref2);
[da.VVdig1] = PVR_getVout(hw.pvr_Vdig1);
[da.VVdig2] = PVR_getVout(hw.pvr_Vdig2);
[da.VVdig3] = PVR_getVout(hw.pvr_Vdig3);
[da.VViol] = PVR_getVout(hw.pvr_Viol);
[da.VVioh] = PVR_getVout(hw.pvr_Vioh);
[da.VVclk] = PVR_getVout(hw.pvr_Vclk);

da.Tenv = PVR_getTemp(hw.pvr_Temp); % enviroument temperature measured by a PVR card

% calculate total power
da.Power = da.VVana*da.IVana+da.VVamp*da.IVamp+da.VVref1*da.IVref1+da.VVref2*da.IVref2+da.VVdig1*da.IVdig1+da.VVdig2*da.IVdig2+da.VVdig3*da.IVdig3;

fprintf('\tANA\t\tAMP\t\tVREF1\t\tVREF2\t\tDIG1\t\tDIG2\t\tDIG3\t|\tCLK\t\tIOL\t\tIOH\t\tIREF1\t\tIREF2\n');
fprintf('V:\t%.2f\t%.2f\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t%.2f\t%.2f\n',da.VVana,da.VVamp,da.VVref1,da.VVref2,da.VVdig1,da.VVdig2,da.VVdig3,da.VVclk,da.VViol,da.VVioh);
fprintf('uA:\t%.0f\t\t%.0f\t\t%.0f\t\t\t%.0f\t\t\t%.0f\t\t\t%.0f\t\t\t%.0f\t\t\t%.0f\t\t%.0f\t\t%.0f\t\t%.2f\t\t%.2f\n',...
    da.IVana*10^6,da.IVamp*10^6,da.IVref1*10^6,da.IVref2*10^6,da.IVdig1*10^6,da.IVdig2*10^6,da.IVdig3*10^6,...
    da.IVclk*10^6,da.IViol*10^6,da.IVioh*10^6,-pr.Iref1,-pr.Iref2);
fprintf('uW:\t%.0f\t\t%.0f\t\t%.0f\t\t\t%.0f\t\t\t%.0f\t\t\t%.0f\t\t\t%.0f\t\t\t%.0f\t\t%.0f\t\t%.0f\n',...
    da.VVana*da.IVana*10^6,da.VVamp*da.IVamp*10^6,da.VVref1*da.IVref1*10^6,da.VVref2*da.IVref2*10^6,da.VVdig1*da.IVdig1*10^6,...
    da.VVdig2*da.IVdig2*10^6,da.VVdig3*da.IVdig3*10^6,da.VVclk*da.IVclk*10^6,da.VViol*da.IViol*10^6,da.VVioh*da.IVioh*10^6);
fprintf('Total uW: \t\t\t\t\t\t\t\t\t\t\t\t\t\t\t%.0f\t\t\t\t\t\t\t%.0f\n',da.Power*10^6,(da.VVclk*da.IVclk+da.VViol*da.IViol+da.VVioh*da.IVioh)*10^6);
fprintf('Temperature: %.1f decC\n\n',da.Tenv);