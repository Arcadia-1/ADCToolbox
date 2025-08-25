function RIGOL_SS_setImpedance(rigol, imp_ch1, imp_ch2)

% imp = 1~10K, 0=high-z

    if(imp_ch1 > 10000 || imp_ch1 < 0 || imp_ch2 > 10000 || imp_ch2 < 0 )
        error('RIGOL-SS: Output impedance out of range (1~10K, 0 for high-Z)');
    end

    RIGOL_modelCheck(rigol, RIGOL_DG4000);
    RIGOL_openConn(rigol);
    
    if(imp_ch1 == 0)
        fprintf(rigol,':OUTPut1:IMPedance INFinity');
    else
        fprintf(rigol,':OUTPut1:IMPedance %d',floor(imp_ch1));
    end
    RIGOL_waitComm;
    
    if(imp_ch2 == 0)
        fprintf(rigol,':OUTPut2:IMPedance INFinity');
    else
        fprintf(rigol,':OUTPut2:IMPedance %d',floor(imp_ch2));
    end
    RIGOL_waitComm;

    fclose(rigol);

end