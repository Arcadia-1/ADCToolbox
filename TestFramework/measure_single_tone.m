if(pr.V_in_comp)
    pr.V_in_peak = pr.V_in_peak_uc * min(1, 0.21 * (pr.F_in_want / pr.F_s).^(-0.75) ) *pr.AVDD_scalling /(1+(pr.single_end==1));
else
    pr.V_in_peak = pr.V_in_peak_uc *pr.AVDD_scalling /(1+(pr.single_end==1));
end

pr.N_bin = findBin(pr.F_s, pr.F_in_want, pr.N_fft);
pr.F_in = pr.N_bin / pr.N_fft * pr.F_s;

Keysight_PSG_set(hw.keysight_psg,pr.F_clk,pr.P_clk);
Keysight_PSG_turn(hw.keysight_psg,1);

pr.T_conv = 1/pr.F_s - pr.T_read;
RIGOL_SS_setPulse(hw.rigol_ss, 1, 3.3, 0, 1/pr.F_s, 0, pr.T_conv, 1*10^-9, 1*10^-9);
RIGOL_SS_setPulse(hw.rigol_ss, 2, 3.3, 0, 1/pr.F_io, 0, 0.5/pr.F_io, 1*10^-9, 1*10^-9);
RIGOL_SS_confBurstExt(hw.rigol_ss, 2, 0, pr.T_conv + pr.T_io_shift, 32);

APX_confSingleEnd(hw.apx,pr.single_end);
APX_setSineCh1(hw.apx, pr.F_in, pr.V_in_peak, pr.V_in_os);

TL413_setSamNum((pr.N_fft+pr.N_ex_sam)*32);

CH341_setGPIO([0,0,1,0,0,0,0,0]);
RIGOL_SS_turn(hw.rigol_ss, 1, 0);
APX_turn(hw.apx,1);

pause(0.1);

da.data_coarse = zeros([pr.N_run, pr.N_fft]);
da.data_fine = zeros([pr.N_run, pr.N_fft]);
da.data_sam = zeros([pr.N_run, pr.N_fft]);

for i_run = 1:pr.N_run
    
    while(1)
    
        TL413_run;
        pause(0.1);
        RIGOL_SS_turn(hw.rigol_ss, 1, 1);

        fprintf('\nStart capturing data for Run #%d/%d, %.1f second(s) left. \n\n',i_run,pr.N_run,(pr.N_fft+pr.N_ex_sam)/pr.F_s*(pr.N_run-i_run+1));
        TL413_wait;

        data = TL413_getData(0)';
        [coarse,fine,sam] = data_phrase(data, pr.N_fft);

        RIGOL_SS_turn(hw.rigol_ss, 1, 0);

        if(sum(15-sam) == 0)
            if(LOOP_MODE)
                data_final = coarse * 32 + fine;
                data_cal = coarse*pr.weight_MSB + fine;
                analysis_single_tone;
                fprintf('\n\nLooping... Press Ctrl+C to exit.\n\n');
                pause(0.1);
            else
                break;
            end
        else
            fprintf('SAM flag is low on some data. May need to extend T_conv. Retrying... \n\n');
        end
    
    end
    
    da.data_coarse(i_run,:) = coarse;
    da.data_fine(i_run,:) = fine;
    da.data_sam(i_run,:) = sam;
    
    fprintf('\n');

end

measure_reference;
record_reference_error;