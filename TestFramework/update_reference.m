PVR_setVout(hw.pvr_Vana, pr.Vana*pr.AVDD_scalling, pr.vtol);
PVR_setVout(hw.pvr_Vamp, pr.Vamp*pr.AVDD_scalling, pr.vtol);
PVR_setVout(hw.pvr_Vref1, pr.Vref1*pr.AVDD_scalling, pr.vtol);
PVR_setVout(hw.pvr_Vref2, pr.Vref2*pr.AVDD_scalling, pr.vtol);
PVR_setVout(hw.pvr_Vdig1, pr.Vdig1, pr.vtol);
PVR_setVout(hw.pvr_Vdig2, pr.Vdig2, pr.vtol);
PVR_setVout(hw.pvr_Vdig3, pr.Vdig3, pr.vtol);
PVR_setVout(hw.pvr_Viol, pr.Viol, pr.vtol);
PVR_setVout(hw.pvr_Vioh, pr.Vioh, pr.vtol);
PVR_setVout(hw.pvr_Vclk, pr.Vclk, pr.vtol);

if(pr.Iref1 <= 0)
    PCR_setIout(hw.pcr_Iref1, pr.Iref1);
else
    PCR_setIout(hw.pcr_Iref1, 0);
    fprintf('!!! >>> REF ERR: Iref1 wrong direction <<< !!!');
end
if(pr.Iref2 <= 0)
    PCR_setIout(hw.pcr_Iref2, pr.Iref2);
else
    PCR_setIout(hw.pcr_Iref2, 0);
    fprintf('!!! >>> REF ERR: Iref2 wrong direction <<< !!!');
end

APX_setVCM(hw.apx,pr.Vcm);  % VCM is provided by the APX audio source