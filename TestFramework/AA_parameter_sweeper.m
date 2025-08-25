% this is for parameter sweeping, it can be use to find the optimal parameter
% it can be also used to perform measurements like DR sweep, Fin sweep, PVT, etc.

if(~exist('da_AIO','var'))
    test_common_head;
end

if(exist('i_bar2','var'))
    i_bar2 = i_bar2 + 1; waitbar(i_bar2/total_bar2, BAR2, sprintf('Measurement (%d/%d)',i_bar2, total_bar2)); 
end

%% Sweeping Settings Library
% "pr.sweep_ref" is the name of parameter to be sweep
% "pr.sweep_list" is the parameter sweeping space

% pr.sweep_ref = 'F_in_want';
% pr.sweep_list = [10.^(-2:0.1:0)*24.9*10^3,(20:5:80)*10^3];
% pr.sweep_list = [0.1,0.5:0.5:24.5,24.9]*10^3;
% pr.sweep_list = [0.1:0.1:5]*10^3;

% pr.sweep_ref = 'F_s';
% pr.sweep_list = [1:50]*10^3;

% pr.sweep_ref = 'F_clk';
% pr.sweep_list = [5:0.1:5.5]*100*10^6;

% pr.sweep_ref = 'V_in_peak_uc';
% pr.sweep_list = [0,10.^(-6:0.5:-0.5)*1,0.9:0.025:1.15];
% pr.sweep_list = [1:0.01:1.1];

% pr.sweep_ref = 'V_in_os';
% pr.sweep_list = [-0.1:0.01:0.1];

% pr.sweep_ref = 'cycGap';
% pr.sweep_list = [0,1,2,3];

% pr.sweep_ref = 'AVDD_scalling';
% pr.sweep_list = [0.9:0.01:1.1];

% pr.sweep_ref = 'Vana';
% pr.sweep_list = [1:0.05:1.4];

% pr.sweep_ref = 'Vamp';
% pr.sweep_list = [0.8:0.05:1.2];

% pr.sweep_ref = 'Vref1';
% pr.sweep_list = [1:0.05:1.4];

% pr.sweep_ref = 'Vref2';
% pr.sweep_list = [0.1:0.1:1.4];

% pr.sweep_ref = 'Vdig1';
% pr.sweep_list = [1.15:0.01:1.2];

% pr.sweep_ref = 'Vcm';
% pr.sweep_list = [0.4:0.025:0.6];

% pr.sweep_ref = 'chopSel';
% pr.sweep_list = [0:7];

% pr.sweep_ref = 'stGnd';
% pr.sweep_list = [0:63];

% pr.sweep_ref = 'osrSel';
% pr.sweep_list = [0:7];

% pr.sweep_ref = 'timing';
% pr.sweep_list = [0:63];

% pr.sweep_ref = 'Iref1';
% pr.sweep_list = -[3:0.5:8];

% pr.sweep_ref = 'Vioh';
% pr.sweep_list = [1:0.1:1.4];


%% Perform Sweeping

% <a similar flow to single tone testing>
pr.test_type = ['Sweep_',pr.sweep_ref];
pr.N_fft =          2^16;
pr.N_run =          1;
pr.F_s =            50 * 10^3;
pr.F_in_want =      5.9 * 10^3;
pr.V_in_peak_uc =   1.05;
pr.V_in_os =        0;

if(exist('pr_AIO','var'))
    pr = overwriteStruct(pr_AIO, pr);
end

parameter_limiter;

pr.V_in_peak = pr.V_in_peak_uc *pr.AVDD_scalling /(1+(pr.single_end==1));
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

N_sweep = length(pr.sweep_list);

da.data_coarse = zeros([N_sweep,pr.N_run, pr.N_fft]);
da.data_fine = zeros([N_sweep,pr.N_run, pr.N_fft]);
da.data_sam = zeros([N_sweep,pr.N_run, pr.N_fft]);

% <extra performance recording vector, define for the specs that you are interested during sweeping
da.SNDR_rec = zeros([1,N_sweep]);
da.SNR_rec = zeros([1,N_sweep]);
da.SFDR_rec = zeros([1,N_sweep]);
da.THD_rec = zeros([1,N_sweep]);
da.amp_rec = zeros([1,N_sweep]);
da.pwr_rec = zeros([1,N_sweep]);
da.FOM_rec = zeros([1,N_sweep]);
da.DC_rec = zeros([1,N_sweep]);
da.Vp_rec = zeros([1,N_sweep]);

% <the sweeping loop>
for i_sweep = 1:N_sweep
    
    fprintf('\nSweeping %s = %.2f  (%d/%d) ...\n\n', pr.sweep_ref, pr.sweep_list(i_sweep), i_sweep, N_sweep);
    
    pr.(pr.sweep_ref) = pr.sweep_list(i_sweep); % change the parameter under sweeping according to the sweeping list
    
    parameter_limiter;  % don't forget to double protect
    
    pr.V_in_peak = pr.V_in_peak_uc *pr.AVDD_scalling /(1+(pr.single_end==1));
    pr.N_bin = findBin(pr.F_s, pr.F_in_want, pr.N_fft);
    pr.F_in = pr.N_bin / pr.N_fft * pr.F_s;
    
    update_reference;   % voltage may needs to be updated
    
    Keysight_PSG_set(hw.keysight_psg,pr.F_clk,pr.P_clk);    % sources may need to be updated
    
    pr.T_conv = 1/pr.F_s - pr.T_read;
    RIGOL_SS_setPulse(hw.rigol_ss, 1, 3.3, 0, 1/pr.F_s, 0, pr.T_conv, 1*10^-9, 1*10^-9);
    RIGOL_SS_setPulse(hw.rigol_ss, 2, 3.3, 0, 1/pr.F_io, 0, 0.5/pr.F_io, 1*10^-9, 1*10^-9);
    RIGOL_SS_confBurstExt(hw.rigol_ss, 2, 0, pr.T_conv + pr.T_io_shift, 32);
    APX_setSineCh1(hw.apx, pr.F_in, pr.V_in_peak, pr.V_in_os);
    
    write_reg(pr,3);                    % configuration may need to be updated
    CH341_setGPIO([0,0,1,0,0,0,0,0]);
    
    pause(1);
    
    % <the same as single test>
    for i_run = 1:pr.N_run
        
        for i_try = 1:5     % there may occure some random errors during sweeping, so give it some retry chance is helpful
                            % put some try-catch if necessary, and don't let your sweeping crash due to some accidental error!

            TL413_run;
            pause(0.5);
            RIGOL_SS_turn(hw.rigol_ss, 1, 1);

            fprintf('\nStart capturing data for Run #%d/%d, %.1f second(s) left. \n\n',i_run,pr.N_run,(pr.N_fft+pr.N_ex_sam)/pr.F_s*(N_sweep*pr.N_run-i_run-pr.N_run*(i_sweep-1)+1));
            TL413_wait;

            data = TL413_getData(0)';
            [coarse,fine,sam] = data_phrase(data, pr.N_fft);

            RIGOL_SS_turn(hw.rigol_ss, 1, 0);

            if(sum(15-sam) == 0)
                break;
            else
                fprintf('SAM flag is low on some data. May need to extend T_conv. Retrying... \n\n');
            end
        
        end

        da.data_coarse(i_sweep,i_run,:) = coarse;
        da.data_fine(i_sweep,i_run,:) = fine;
        da.data_sam(i_sweep,i_run,:) = sam;

    end
    
    % <analyze the data of each swept testpoint>
    fprintf('Analyzing Data for %s = %.2f  ...\n\n', pr.sweep_ref, pr.sweep_list(i_sweep));

    data_final = reshape(da.data_coarse(i_sweep,:,:) * 32 + da.data_fine(i_sweep,:,:),[1,pr.N_fft*pr.N_run]);
    data_cal = reshape(da.data_coarse(i_sweep,:,:) * pr.weight_MSB + da.data_fine(i_sweep,:,:),[1,pr.N_fft*pr.N_run]);

    analysis_single_tone;
    
    measure_reference;
    
    % <record the analysis results in the vector>
    da.Vp_rec(i_sweep) = pr.V_in_peak;
    da.SNDR_rec(i_sweep) = SNDR;
    da.SNR_rec(i_sweep) = SNR;
    da.SFDR_rec(i_sweep) = SFDR;
    da.THD_rec(i_sweep) = THD;
    da.amp_rec(i_sweep) = amp;
    da.pwr_rec(i_sweep) = da.VVana*da.IVana+da.VVamp*da.IVamp+da.VVref1*da.IVref1+da.VVref2*da.IVref2+da.VVdig1*da.IVdig1+da.VVdig2*da.IVdig2+da.VVdig3*da.IVdig3;
    da.FOM_rec(i_sweep) = SNDR+10*log10(pr.F_s/2/da.pwr_rec(i_sweep));
    da.DC_rec(i_sweep) = mean(data_cal) / (33*2^(pr.osrSel+7)) * 2 - 1;
end

record_reference_error;


%% Results Plot
% plot any X vs Y figure as interested

figure(5);
clf;
hold on;
plot(pr.sweep_list,da.SNDR_rec,'linewidth',2);
plot(pr.sweep_list,da.SNR_rec,'linewidth',2);
plot(pr.sweep_list,da.SFDR_rec,'linewidth',2);
plot(pr.sweep_list,-da.THD_rec,'linewidth',2);
plot(pr.sweep_list,da.SNR_rec-da.amp_rec,'linewidth',2);
ylabel('dB');
xlabel(pr.sweep_ref);
grid on;
legend('SNDR','SNR','SFDR','-THD','Noise');

figure(6);
plot(pr.sweep_list,da.amp_rec,'linewidth',2);
ylabel('dB');
xlabel(pr.sweep_ref);
grid on;
legend('AMP');

figure(7);
plot(pr.sweep_list,da.pwr_rec*10^6,'linewidth',2);
grid on;
xlabel(pr.sweep_ref);
ylabel('uW');
legend('Power');

figure(8);
plot(pr.sweep_list,da.FOM_rec,'linewidth',2);
grid on;
xlabel(pr.sweep_ref);
ylabel('dB');
legend('FoMs');

figure(9);
plot(pr.sweep_list,da.DC_rec,'linewidth',2);
grid on;
xlabel(pr.sweep_ref);
ylabel('Full Scale');
legend('Normalized DC');


[FOMs,i_best] = max(da.FOM_rec);
da.SNDR = da.SNDR_rec(i_best);
da.SNR = da.SNR_rec(i_best);
da.SFDR = da.SFDR_rec(i_best);
da.THD = da.THD_rec(i_best);
da.Signal = da.amp_rec(i_best);
da.Noise = da.SNR_rec(i_best)-da.amp_rec(i_best);
da.Power = da.pwr_rec(i_best);
da.DC = da.DC_rec(i_best);
da.FOMs = FOMs;
da.FOMw = da.Power/pr.F_s/2^((SNDR-1.76)/6.02);


%%
test_common_tail;