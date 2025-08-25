% <initialize. for all_in_one test / sweeping, no need to init>
if(~exist('da_AIO','var'))  
    test_common_head;
end

% <wait bar for all_in_one test, optional>
if(exist('i_bar2','var'))
    i_bar2 = i_bar2 + 1; waitbar(i_bar2/total_bar2, BAR2, sprintf('Measurement (%d/%d)',i_bar2, total_bar2)); 
end

%%
% <flags of debugging modes>
LOOP_MODE = 0;                  % loop the test until ctrl-C ,but does not record results (for live debug)

% <parameters of testing>
pr.test_type = 'SingleTone';    
pr.N_fft =          2^16;
pr.N_run =          16;             % multiple run for spectrum averaging
pr.F_s =            50 * 10^3;      % sampling frequency
pr.F_in_want =      5.9 * 10^3;     % expected input frequency
pr.V_in_peak_uc =   1.1;            % magnitude of input sinewave
pr.V_in_os =        0;              % offset input sinewave

% <for parameter sweeping, overwrite the parameters that specified in pr_AIO>
if(exist('pr_AIO','var'))       
    pr = overwriteStruct(pr_AIO, pr);
end

% <a double check of the key parameters, e.g., supply voltage, don't overvotage!>
parameter_limiter;    

% <the derived parameters>
pr.V_in_peak = pr.V_in_peak_uc *pr.AVDD_scalling /(1+(pr.single_end==1));   % actual input swing
pr.N_bin = findBin(pr.F_s, pr.F_in_want, pr.N_fft);     
pr.F_in = pr.N_bin / pr.N_fft * pr.F_s;                 % actual input frequency (coherent)

% <config supplies & bias voltage/current>
update_reference;         

% <config RF clock source>
Keysight_PSG_set(hw.keysight_psg,pr.F_clk,pr.P_clk);
Keysight_PSG_turn(hw.keysight_psg,1);

% <config AWG clock source>
pr.T_conv = 1/pr.F_s - pr.T_read;
RIGOL_SS_setPulse(hw.rigol_ss, 1, 3.3, 0, 1/pr.F_s, 0, pr.T_conv, 1*10^-9, 1*10^-9);     % sampling clk 
RIGOL_SS_setPulse(hw.rigol_ss, 2, 3.3, 0, 1/pr.F_io, 0, 0.5/pr.F_io, 1*10^-9, 1*10^-9);  % IO clk
RIGOL_SS_confBurstExt(hw.rigol_ss, 2, 0, pr.T_conv + pr.T_io_shift, 32);

% <config signal source>
APX_confSingleEnd(hw.apx,pr.single_end);            
APX_setSineCh1(hw.apx, pr.F_in, pr.V_in_peak, pr.V_in_os);

% <config logic analyzer>
TL413_setSamNum((pr.N_fft+pr.N_ex_sam)*32);           % capture length

% <config chip - SPI, I2C, conf bits, etc.>
write_reg(pr,3);                      
CH341_setGPIO([0,0,1,0,0,0,0,0]);

% <turn on sampling clk and signal source>
RIGOL_SS_turn(hw.rigol_ss, 1, 1); 
APX_turn(hw.apx,1);

% <wait for equipment and chip settling>
pause(0.1); 

% <init data recording vector>
da.data_coarse = zeros([pr.N_run, pr.N_fft]);
da.data_fine = zeros([pr.N_run, pr.N_fft]);
da.data_sam = zeros([pr.N_run, pr.N_fft]);

% <main testing loop, for multiple run>
for i_run = 1:pr.N_run
    
    while(1)    % for LOOP_MODE
    
        % <capture data using logic analyzer>
        TL413_run;
        fprintf('\nStart capturing data for Run #%d/%d, %.1f second(s) left. \n\n',i_run,pr.N_run,(pr.N_fft+pr.N_ex_sam)/pr.F_s*(pr.N_run-i_run+1));
        TL413_wait;                 % wait till end of capture
        data = TL413_getData(0)';   % retrive data from logic analyzer
                                    % to get data of mutiple ch, getDataBus can be used
        
        % <parse data for easier analyzer, optional>
        [coarse,fine,sam] = data_parse(data, pr.N_fft); % sam is the flag for successful conversion

        % <verify if the data is valid>
        if(sum(15-sam) == 0)
            if(LOOP_MODE)   % in LOOP_MODE, loopback and do realtime analysis
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
    
    % <record interested data>
    da.data_coarse(i_run,:) = coarse;
    da.data_fine(i_run,:) = fine;
    da.data_sam(i_run,:) = sam;
    
    fprintf('\n');

end

% <measure supply / bias / power>
measure_reference;
record_reference_error; % record the error flags as well for future review, this is optional

%%
fprintf('Analyzing Data...\n\n');

% <data post processing>
data_final = da.data_coarse * 32 + da.data_fine;
data_cal = da.data_coarse*pr.weight_MSB + da.data_fine;

% <data analyzer>
analysis_single_tone;

% <record interested results>
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
% <common procedures for ending a test>
test_common_tail;