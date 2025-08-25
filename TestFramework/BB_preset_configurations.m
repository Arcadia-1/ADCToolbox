% configurations below are related to your actual testing plan

pr.stGnd = bin2dec('1 1 1 0 0 1');
pr.refCap = 1;
pr.clkInBuf = 1;
pr.enOsc = 0;
pr.clkSel = 1;  % 0: internal, 1: external
pr.clkOut = 1;
pr.outMode = 0; % 0: push-pull, 1: push only
pr.refStb = 1;
pr.prbSel = 1;  % 0: low, 1: RDY, 2: DAC_RLST, 3: DAC_SET, 4:COM_CHOP, 5:COM_RST, 6:DAC_ON, 7:CNT_LOCAL[6]
pr.osrSel = 6;  % 0: 1x, 1: 2x, 2: 4x, 3: 8x, 4: 16x, 5: 32x, 6: 64x, 7: 128x
pr.chopSel = 7; % 0: no chop, 1: p64, 2: p32, 3: p16, 4: p8, 5: p4, 6: p2, 7: p1
pr.timing = bin2dec('11 01 00');
pr.cycGap = 0;  % 0: 4 cycles, 1: 8 cycles, 2: 16 cycles, 3: 32 cycles
pr.enDel = 1;   % turn on if one of [dacDel,firDel,ddbWid] >0
pr.enRot = 1;

write_reg(pr,3);    % config the chip