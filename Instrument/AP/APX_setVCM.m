function APX_setVCM(apx,vcm)

%     if(vcm < -1 || vcm > 1)
%         error('APX: VCM setting out of range (-1~1V)');
%     end

    apx.BenchMode.Setup.AdcTest.VBias = vcm;

end