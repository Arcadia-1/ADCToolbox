% an example of two tone testing, similar flow as single tone test

if(~exist('da_AIO','var'))  
    test_common_head;
end

if(exist('i_bar2','var'))
    i_bar2 = i_bar2 + 1; waitbar(i_bar2/total_bar2, BAR2, sprintf('Measurement (%d/%d)',i_bar2, total_bar2));  
end

%%
LOOP_MODE = 0;

pr.test_type = 'TwoTone';

pr.N_fft =          2^16;
pr.N_run =          16;
pr.F_s =            50 * 10^3;
pr.F_in_want1 =     0.9 * 10^3;
pr.F_in_want2 =     1 * 10^3;
pr.V_in_peak_uc =   1.05;
pr.V_in_os =        0;

if(exist('pr_AIO','var'))
    pr = overwriteStruct(pr_AIO, pr);
end

parameter_limiter;   

pr.V_in_peak = pr.V_in_peak_uc *pr.AVDD_scalling /(1+(pr.single_end==1));
pr.N_bin1 = findBin(pr.F_s, pr.F_in_want1, pr.N_fft);
pr.N_bin2 = findBin(pr.F_s, pr.F_in_want2, pr.N_fft);
pr.F_in1 = pr.N_bin1 / pr.N_fft * pr.F_s;
pr.F_in2 = pr.N_bin2 / pr.N_fft * pr.F_s;

update_reference;   

Keysight_PSG_set(hw.keysight_psg,pr.F_clk,pr.P_clk);
Keysight_PSG_turn(hw.keysight_psg,1);

pr.T_conv = 1/pr.F_s - pr.T_read;
RIGOL_SS_setPulse(hw.rigol_ss, 1, 3.3, 0, 1/pr.F_s, 0, pr.T_conv, 1*10^-9, 1*10^-9);
RIGOL_SS_setPulse(hw.rigol_ss, 2, 3.3, 0, 1/pr.F_io, 0, 0.5/pr.F_io, 1*10^-9, 1*10^-9);
RIGOL_SS_confBurstExt(hw.rigol_ss, 2, 0, pr.T_conv + pr.T_io_shift, 32);

APX_setDualSineCh1(hw.apx, pr.F_in1,pr.F_in2, pr.V_in_peak, pr.V_in_os);

TL413_setSamNum((pr.N_fft+pr.N_ex_sam)*32);

write_reg(pr,3);
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
        fprintf('\nStart capturing data for Run #%d/%d, %.1f second(s) left. \n\n',i_run,pr.N_run,(pr.N_fft+pr.N_ex_sam)/pr.F_s*(pr.N_run-i_run+1));
        TL413_wait;

        data = TL413_getData(0)';
        [coarse,fine,sam] = data_phrase(data, pr.N_fft);

        RIGOL_SS_turn(hw.rigol_ss, 1, 0);

        if(sum(15-sam) == 0)
            if(LOOP_MODE)
                data_final = coarse * 32 + fine;
                data_cal = coarse*pr.weight_MSB + fine;
                analysis_two_tone;
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


%%
fprintf('Analyzing Data...\n\n');

data_final = da.data_coarse * 32 + da.data_fine;
data_cal = da.data_coarse*pr.weight_MSB + da.data_fine;

analysis_two_tone;

da.SNDR = SNDR;
da.ENOB = ENOB;
da.SNR = SNR;
da.SFDR = SFDR;
da.THD = THD;
da.Signal = amp;
da.Noise = SNR-amp;
da.DC = mean(data_cal) / (33*2^(pr.osrSel+7)) * 2 - 1;
da.FOMs = SNDR+10*log10(pr.F_s/2/da.Power);
da.FOMw = da.Power/pr.F_s/2^((SNDR-1.76)/6.02);

fprintf('\n\tSNDR\t\tENOB\tSNR\t\t\tSFDR\t\tTHD\t\t\t\tFOMs\t\t\tFOMw\t\t\tP/Fs\n');
fprintf('\t%.1fdB\t\t%.1f\t%.1fdB\t\t%.1fdB\t\t%.1fdB\t\t%.1fdB\t\t\t%.1ffJ\t\t%.1fpJ\n\n',SNDR,ENOB,SNR,SFDR,THD,da.FOMs,da.FOMw*10^15,da.Power/pr.F_s*10^12);



%%
test_common_tail;