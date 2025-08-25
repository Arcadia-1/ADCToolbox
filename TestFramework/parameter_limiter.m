% write your protection rules here
% be really careful on this part!! 
% especially for supply voltage and current direction

pr.Vana = max(min(pr.Vana,1.4),0.1);
pr.Vamp = max(min(pr.Vamp,1.4),0.1);
pr.Vref1 = max(min(pr.Vref1,1.4),0.1);
pr.Vref2 = max(min(pr.Vref2,1.4),0.1);
pr.Vdig1 = max(min(pr.Vdig1,1.4),0.1);
pr.Vdig2 = max(min(pr.Vdig2,1.4),0.1);
pr.Vdig3 = max(min(pr.Vdig3,1.4),0.1);
pr.Viol = max(min(pr.Viol,1.4),0.1);
pr.Vioh = max(min(pr.Vioh,1.4),0.1);
pr.Vclk = max(min(pr.Vclk,1.4),0.1);
pr.Vcm = max(min(pr.Vcm,1.4),0.1);

pr.Iref1 = max(min(pr.Iref1,0),-10);
pr.Iref2 = max(min(pr.Iref2,0),-10);

pr.stGnd = mod(pr.stGnd,64);
pr.refCap = mod(pr.refCap,2);
pr.clkInBuf = mod(pr.clkInBuf,2);
pr.enOsc = mod(pr.enOsc,2);
pr.clkSel = mod(pr.clkSel,2);  % 0: internal, 1: external
pr.clkOut = mod(pr.clkOut,2);
pr.outMode = mod(pr.outMode,2); % 0: push-pull, 1: push only
pr.refStb = mod(pr.refStb,2);
pr.prbSel = mod(pr.prbSel,8);  % 0: low, 1: RDY, 2: DAC_RLST, 3: DAC_SET, 4:COM_CHOP, 5:COM_RST, 6:DAC_ON, 7:CNT_LOCAL[6]
pr.osrSel = mod(pr.osrSel,8);  % 0: 1x, 1: 2x, 2: 4x, 3: 8x, 4: 16x, 5: 32x, 6: 64x, 7: 128x
pr.chopSel = mod(pr.chopSel,8); % 0: no chop, 1: p64, 2: p32, 3: p16, 4: p8, 5: p4, 6: p2, 7: p1
pr.timing = mod(pr.timing,64);
pr.cycGap = mod(pr.cycGap,4);  % 0: 4 cycles, 1: 8 cycles, 2: 16 cycles, 3: 32 cycles
pr.enDel = pr.timing > 0;   % turn on if one of [dacDel,firDel,ddbWid] >0
pr.enRot = mod(pr.enRot,2);


pr.F_clk = max(min(pr.F_clk,1000*10^6),100*10^6);
pr.P_clk = max(min(pr.P_clk,0),-30); 

pr.T_read = max(min(pr.T_read,10*10^-6),1*10^-6);

pr.F_io = max(min(pr.F_io,30*10^6),1*10^6);
pr.T_io_shift = max(min(pr.T_io_shift,1*10^-6),-1*10^-6);

pr.V_in_comp = mod(pr.V_in_comp,2);

pr.AVDD_scalling = max(min(pr.AVDD_scalling,1.2),0.8);

pr.single_end = mod(pr.single_end,2);