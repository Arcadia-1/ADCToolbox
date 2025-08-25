% this is an example flow of "all in one test" for a new chip
% it contains basic funtional test, parameter automatic optimization, and
% perform the final performance test and sweepings, all data are
% automatically saved

% please read AA_test_single_tone.m and AA_parameter_sweeper.m before go through this one

%% Initialize

close all
clc;

if(exist('hw','var'))
    try
        fprintf('Shutting Down ...\n\n');
        AA_shutdown;
    catch
    end
    pause(5);
end

clear all

MAX_TRY = 5;    % maximum retry time
TOTAL_ITEM = 8; % all in one test items

path_setting;

fprintf('Init Hardware ...\n\n');
init_hardware;

pr = struct();  % parameters
da = struct();  % data
da_AIO = struct();  % master parameter set

% <default parameters for all in one test>
BB_default_common_settings;
BB_default_reference;
BB_default_configurations;
pr_AIO = pr;

% <turn on supplies>
clear_reference_error;
update_reference;
turnon_reference;

fprintf('Waiting Vref Stable ...\n\n');
pause(5);

% <record test start time>
clc;
fprintf('Test Start ...\n\n');
da_AIO.time_start = datetime;

% <create a waitbar to show test progress>
BAR = waitbar(0,sprintf('All in One Test (0 / %d)',TOTAL_ITEM));
BAR.Position = [0,50,300,50];

%% ----------------------------------------------------------------------------------------

pr_AIO.chip_id = 26;    % don't forget to change this when test a new chip!!!!!!!!!!!!!!!!!!!!!!!!!!


%% the following test items are from your test plan
%% Reg W/R Test
% test if the chip can be config normally

fprintf('\n\n??? Testing Reg W/R... \n\n');
waitbar(1/TOTAL_ITEM, BAR, sprintf('All in One Test (1 / %d): Reg W/R',TOTAL_ITEM));    % update the waitbar as testing going
dist_win;   % show the live monitors (figures)

SUCC = 0;
for itry = 1:MAX_TRY
    write_reg(pr_AIO,0);
    pause(0.1);
    if(verify_reg(pr_AIO,1))
        SUCC = 1;
        break;
    end
    pause(0.1);
end
if(~SUCC)
    da_AIO.err = 'Reg W/R Failure';
    test_fail;
end
da_AIO.succ_reg = 1;
fprintf('\n\n!!! Reg W/R Passed! \n\n');

pause(5);

close all;
clc;

%% check volt and current
% test if the supply current is within normal range

fprintf('\n\n??? Testing Supply Current ... \n\n');
waitbar(2/TOTAL_ITEM, BAR, sprintf('All in One Test (2 / %d): Supply Current',TOTAL_ITEM));
dist_win;

pr_AIO.N_fft =          2^10;
pr_AIO.N_run =          1;
pr_AIO.F_s =            10 * 10^3;
pr_AIO.F_in_want =      1 * 10^3;
pr_AIO.V_in_peak_uc =   0;
pr_AIO.V_in_os =        0;

AA_test_single_tone;    % run single test for once to get supply current data

if(da.IVana < 1*10^-6 || da.IVamp < 300*10^-6 || da.IVref1 < 10*10^-6 || da.IVref2 < 1*10^-6 || da.IVdig1 < 30*10^-6 || da.IVdig2 < 10*10^-6 || da.IVdig3 < 30*10^-6 || da.IVclk < 100*10^-6)
    da_AIO.err = 'Supply Current Exception';
    test_fail;
end
da_AIO.succ_supply = 1;
fprintf('\n\n!!! Supply Current Checking Passed! \n\n');

pause(5);

close all;
clc;


%% IO speed: Sam, the last two bit in input short
% check if the IO can run at full rate, and try up-tuning the IO supply voltage to make it fast enough

fprintf('\n\n??? Testing Data IO ... \n\n');
waitbar(3/TOTAL_ITEM, BAR, sprintf('All in One Test (3 / %d): Data IO',TOTAL_ITEM));
dist_win;

pr_AIO.N_fft =          2^12;
pr_AIO.N_run =          1;
pr_AIO.F_s =            10 * 10^3;
pr_AIO.F_in_want =      1 * 10^3;
pr_AIO.V_in_peak_uc =   0.9;
pr_AIO.V_in_os =        0;
pr_AIO.T_read =         2*10^-6;
pr_AIO.F_io =           20*10^6;

pr_AIO.sweep_ref = 'Vioh';          % sweep IO voltage
pr_AIO.sweep_list = [1:0.05:1.4];

AA_parameter_sweeper;

pass_list = (abs(da.DC_rec) < 0.05) & (da.SNR_rec-da.amp_rec > 80);  % judge if each test point in the sweep is pass or failed
pass_value = pr_AIO.sweep_list(pass_list);
if(~isempty(pass_value))
    pr_AIO.Vioh = pass_value(min(3,length(pass_value)));    % pick a suitable IO voltage
else
    da_AIO.err = 'Data IO Failed';  % this is the reason for failure
    test_fail;      % call this when a AIO test failed somehow
end
da_AIO.succ_io = 1;
fprintf('\n\n!!! Data IO Checking Passed! (Vioh = %.2fV)\n\n',pr_AIO.Vioh);

pause(5);

close all;
clc;

%% noise floor optimizing
% optimize the input shorted noise floor by turning some paramteres

fprintf('\n\n??? Optimizing Noise Floor ... \n\n');
waitbar(4/TOTAL_ITEM, BAR, sprintf('All in One Test (4 / %d): Optimizing Noise Floor',TOTAL_ITEM));
dist_win;

pr_AIO.N_fft =          2^16;
pr_AIO.N_run =          1;
pr_AIO.F_s =            20 * 10^3;
pr_AIO.F_in_want =      1 * 10^3;
pr_AIO.V_in_peak_uc =   0.02;
pr_AIO.V_in_os =        0;
pr_AIO.F_clk = 500*10^6;
pr_AIO.Vref2 = 0.1;
pr_AIO.Iref1 = -5;

SUCC = 0;
for i_try = 1:MAX_TRY

    AA_test_single_tone;

    sweep_MSB_weight;

    if(max(SNDR_rec)-min(SNDR_rec) > 1)
        SUCC = 1;
        break;
    end
end

if(SUCC == 0)
    da_AIO.err = 'Weight Search Failed';
    test_fail;
end

[~,i_best] = max(SNDR_rec);
pr_AIO.weight_MSB = gain_list(i_best);

% sweep other parameters
pr_AIO.V_in_peak_uc =   0.02;
pr_AIO.F_clk = 4*100*10^6;
pr_AIO.Vref2 = 0.9;
pr_AIO.Iref1 = -5;
pr_AIO.chopSel = 7;
OPT_LIST = {                        % task list of optimization, each steps of optimization contains a parameter to be opt and a range to be search
    'F_clk',[0:0.5:4]*100*10^6,
    'Vref2',[-0.8:0.1:0.2],
    'Iref1',[2:-0.5:-3],
    'F_clk',[-1:0.1:1]*100*10^6,
    'Vref2',[-0.2:0.025:0.2],
    'Iref1',[-1.5:0.25:1.5],
    'chopSel',[0:-1:-7],
    }';
OPT_REC = zeros([length(OPT_LIST),7]);

pr_AIO.assumedSignal = -1;

BAR2 = waitbar(0,'');
BAR2.Position = [0,120,300,50];

for i_opt = 1:length(OPT_LIST)  % iterate through the OPT_LIST and do sweeping (linear search) for each optimization step
    
    waitbar(i_opt/length(OPT_LIST),BAR2,sprintf('Optimizing %s (%d/%d) ...',OPT_LIST{i_opt*2-1},i_opt,length(OPT_LIST)));
    
    pr_AIO.sweep_ref = OPT_LIST{i_opt*2-1};
    pr_AIO.sweep_list = pr_AIO.(pr_AIO.sweep_ref) + OPT_LIST{i_opt*2};
    
    AA_parameter_sweeper;
    
    pass_DC = da.DC_rec;
    pass_FOM = da.FOM_rec;
    pass_id = 1:length(pass_DC);

    pass_list = (abs(pass_DC) < 0.05) & (pass_FOM > 150) & (pass_FOM < 190);
    pass_id = pass_id(pass_list);
    pass_para = pr_AIO.sweep_list(pass_list);
    pass_DC = pass_DC(pass_list);
    pass_FOM = pass_FOM(pass_list);
    
    if(~isempty(pass_para)) % if find a better valid parameter, change the current one to it.
        [~,i_best] = max(pass_FOM);
        pr_AIO.(pr_AIO.sweep_ref) = pass_para(i_best);
        
        fprintf('\n\n>>> Set %s to %e \n\n',pr_AIO.sweep_ref,pass_para(i_best));
        
        OPT_REC(i_opt,1:7) = [
            da.SNDR_rec(pass_id(i_best)),
            da.SNR_rec(pass_id(i_best)),
            da.SFDR_rec(pass_id(i_best)),
            da.THD_rec(pass_id(i_best)),
            da.amp_rec(pass_id(i_best)),
            da.pwr_rec(pass_id(i_best)),
            da.FOM_rec(pass_id(i_best))];
        
        figure(101);
        clf;
        plot(OPT_REC(1:i_opt,7),'linewidth',2);
        ylabel('FOMS (dB)');
        xlabel('Iteration');
    else
        fprintf('\n\n>>> %s not changed \n\n',pr_AIO.sweep_ref);    % failed to find a better valid parameter, so keep the current value
    end
    
end

pr_AIO.assumedSignal = NaN;

da_AIO.succ_noisefloor = 1;
fprintf('\n\n!!! Noise Floor Optimization Complete! (FOM = %.2fdB)\n\n',pass_FOM(i_best));

da_AIO.pr_opt_noise_floor = pr_AIO;
da_AIO.opt_noise_floor = OPT_REC;

pause(5);

close(BAR2);
close all;
clc;

%% large signal optimizing: low bw case
% similar to noise floor optimization

fprintf('\n\n??? Optimizing Low Fs Mode ... \n\n');
waitbar(5/TOTAL_ITEM, BAR, sprintf('All in One Test (5 / %d): Optimizing Low Fs Mode',TOTAL_ITEM));
dist_win;

pr_AIO = da_AIO.pr_opt_noise_floor;

pr_AIO.N_fft =          2^16;
pr_AIO.N_run =          1;
pr_AIO.F_s =            20 * 10^3;
pr_AIO.F_in_want =      0.9 * 10^3;
pr_AIO.V_in_peak_uc =   1;
pr_AIO.V_in_os =        0;
pr_AIO.osrSel =         7;  


OPT_LIST = {
    'F_clk',[-1:0.5:4]*100*10^6,
    'Iref1',[-3:0.5:1],
    'Vana',[0:-0.1:-0.5],
    'Vdig1',[0.1:-0.1:-0.3],
    'Vdig3',[0.1:-0.1:-0.3],
    'Vref1',[0.1:-0.1:-0.3],
    
    'F_clk',[-1:0.1:1]*100*10^6,
    'Vref2',[-0.2:0.025:0.2],
    'Vamp',[-0.2:0.1:0.2],
    
    'chopSel',[0:1:7],
    'osrSel',[0:1:7],
    'cycGap',[0:1:3],
    'timing',[0:1:63],
    'stGnd',[0:1:63],
    
    'Vcm',[-0.3:0.05:0.3],
    'V_in_os',[-0.1:0.025:0.1],
    'V_in_peak_uc',[-0.2:0.025:0.2],
    
    'refStb',[0,1],
    
    'Vana',[-0.2:0.025:0.2],
    'Vref1',[-0.2:0.025:0.2],
    
    'refStb',[0,1],
    
    'Vana',[-0.2:0.025:0.2],
    'Vref1',[-0.2:0.025:0.2],
    'Vdig1',[-0.2:0.025:0.2],
    'Vdig3',[-0.2:0.025:0.2],
    
    'F_clk',[-1:0.1:1]*100*10^6,
    'Vref2',[-0.2:0.025:0.2],
    'Iref1',[-1.5:0.25:1.5],
    'Vamp',[-0.2:0.025:0.2],
    
    'Vdig2',[-0.2:0.025:0.2],
    
    'cycGap',[0:1:3],
    'timing',[0:1:63],
    'stGnd',[0:1:63],
    
    'Vcm',[-0.3:0.05:0.3],
    'V_in_os',[-0.1:0.025:0.1],
    'V_in_peak_uc',[-0.2:0.025:0.2],
    
    }';

OPT_REC = zeros([length(OPT_LIST),7]);

BAR2 = waitbar(0,'');
BAR2.Position = [0,120,300,50];

for i_opt = 1:length(OPT_LIST)
    
    waitbar(i_opt/length(OPT_LIST),BAR2,sprintf('Optimizing %s (%d/%d) ...',OPT_LIST{i_opt*2-1},i_opt,length(OPT_LIST)));
    
    pr_AIO.sweep_ref = OPT_LIST{i_opt*2-1};
    pr_AIO.sweep_list = pr_AIO.(pr_AIO.sweep_ref) + OPT_LIST{i_opt*2};
    
    AA_parameter_sweeper;
    
    pass_DC = da.DC_rec;
    pass_FOM = da.FOM_rec;
    pass_id = 1:length(pass_DC);

    pass_list = (abs(pass_DC) < 0.15) & (pass_FOM > 150) & (pass_FOM < 190);
    pass_id = pass_id(pass_list);
    pass_para = pr_AIO.sweep_list(pass_list);
    pass_DC = pass_DC(pass_list);
    pass_FOM = pass_FOM(pass_list);
    
    if(~isempty(pass_para))
        [~,i_best] = max(pass_FOM);
        pr_AIO.(pr_AIO.sweep_ref) = pass_para(i_best);
        
        fprintf('\n\n>>> Set %s to %e \n\n',pr_AIO.sweep_ref,pass_para(i_best));
        
        OPT_REC(i_opt,1:7) = [
            da.SNDR_rec(pass_id(i_best)),
            da.SNR_rec(pass_id(i_best)),
            da.SFDR_rec(pass_id(i_best)),
            da.THD_rec(pass_id(i_best)),
            da.amp_rec(pass_id(i_best)),
            da.pwr_rec(pass_id(i_best)),
            da.FOM_rec(pass_id(i_best))];
        
        figure(101);
        clf;
        plot(OPT_REC(1:i_opt,7),'linewidth',2);
        ylabel('FOMS (dB)');
        xlabel('Iteration');
    else
        fprintf('\n\n>>> %s not changed \n\n',pr_AIO.sweep_ref);
    end
    
end

da_AIO.succ_low_fs = 1;
fprintf('\n\n!!! Low Fs Optimization Complete! (FOM = %.2fdB)\n\n',pass_FOM(i_best));

da_AIO.pr_opt_low_fs = pr_AIO;
da_AIO.opt_low_fs = OPT_REC;

pause(5);

close(BAR2);
close all;
clc;

%% large signal optimizing: high bw case
% similar to noise floor optimization

fprintf('\n\n??? Optimizing High Fs Mode ... \n\n');
waitbar(6/TOTAL_ITEM, BAR, sprintf('All in One Test (6 / %d): Optimizing High Fs Mode',TOTAL_ITEM));
dist_win;

% pr_AIO = da_AIO.pr_opt_noise_floor;

pr_AIO.N_fft =          2^16;
pr_AIO.N_run =          1;
pr_AIO.F_s =            50 * 10^3;
pr_AIO.F_in_want =      0.9 * 10^3;
pr_AIO.V_in_peak_uc =   1;
pr_AIO.V_in_os =        0;
pr_AIO.osrSel =         6;  


OPT_LIST = {
    'F_clk',[-1:0.5:4]*100*10^6,
    'Iref1',[-3:0.5:1],
    'Vana',[0:-0.1:-0.5],
    'Vdig1',[0.1:-0.1:-0.3],
    'Vdig3',[0.1:-0.1:-0.3],
    'Vref1',[0.1:-0.1:-0.3],
    
    'F_clk',[-1:0.1:1]*100*10^6,
    'Vref2',[-0.2:0.025:0.2],
    'Iref1',[-1.5:0.25:1.5],
    'Vamp',[-0.2:0.1:0.2],
    
    'chopSel',[0:1:7],
    'osrSel',[0:1:7],
    'cycGap',[0:1:3],
    'timing',[0:1:63],
    'stGnd',[0:1:63],
    
    'Vcm',[-0.3:0.05:0.3],
    'V_in_os',[-0.1:0.025:0.1],
    'V_in_peak_uc',[-0.2:0.025:0.2],
    
    'refStb',[0,1],
    
    'Vana',[-0.2:0.025:0.2],
    'Vref1',[-0.2:0.025:0.2],
    
    'refStb',[0,1],
    
    'Vana',[-0.2:0.025:0.2],
    'Vref1',[-0.2:0.025:0.2],
    'Vdig1',[-0.2:0.025:0.2],
    'Vdig3',[-0.2:0.025:0.2],
    
    'F_clk',[-1:0.1:1]*100*10^6,
    'Vref2',[-0.2:0.025:0.2],
    'Iref1',[-1.5:0.25:1.5],
    'Vamp',[-0.2:0.025:0.2],
    
    'Vdig2',[-0.2:0.025:0.2],
    
    'cycGap',[0:1:3],
    'timing',[0:1:63],
    'stGnd',[0:1:63],
    
    'Vcm',[-0.3:0.05:0.3],
    'V_in_os',[-0.1:0.025:0.1],
    'V_in_peak_uc',[-0.2:0.025:0.2],
    
    }';

OPT_REC = zeros([length(OPT_LIST),7]);

BAR2 = waitbar(0,'');
BAR2.Position = [0,120,300,50];

for i_opt = 1:length(OPT_LIST)
    
    waitbar(i_opt/length(OPT_LIST),BAR2,sprintf('Optimizing %s (%d/%d) ...',OPT_LIST{i_opt*2-1},i_opt,length(OPT_LIST)));
    
    pr_AIO.sweep_ref = OPT_LIST{i_opt*2-1};
    pr_AIO.sweep_list = pr_AIO.(pr_AIO.sweep_ref) + OPT_LIST{i_opt*2};
    
    AA_parameter_sweeper;
    
    pass_DC = da.DC_rec;
    pass_FOM = da.FOM_rec;
    pass_id = 1:length(pass_DC);

    pass_list = (abs(pass_DC) < 0.15) & (pass_FOM > 150) & (pass_FOM < 190);
    pass_id = pass_id(pass_list);
    pass_para = pr_AIO.sweep_list(pass_list);
    pass_DC = pass_DC(pass_list);
    pass_FOM = pass_FOM(pass_list);
    
    if(~isempty(pass_para))
        [~,i_best] = max(pass_FOM);
        pr_AIO.(pr_AIO.sweep_ref) = pass_para(i_best);
        
        fprintf('\n\n>>> Set %s to %e \n\n',pr_AIO.sweep_ref,pass_para(i_best));
        
        OPT_REC(i_opt,1:7) = [
            da.SNDR_rec(pass_id(i_best)),
            da.SNR_rec(pass_id(i_best)),
            da.SFDR_rec(pass_id(i_best)),
            da.THD_rec(pass_id(i_best)),
            da.amp_rec(pass_id(i_best)),
            da.pwr_rec(pass_id(i_best)),
            da.FOM_rec(pass_id(i_best))];
        
        figure(101);
        clf;
        plot(OPT_REC(1:i_opt,7),'linewidth',2);
        ylabel('FOMS (dB)');
        xlabel('Iteration');
    else
        fprintf('\n\n>>> %s not changed \n\n',pr_AIO.sweep_ref);
    end
    
end

da_AIO.succ_low_fs = 1;
fprintf('\n\n!!! High Fs Optimization Complete! (FOM = %.2fdB)\n\n',pass_FOM(i_best));

da_AIO.pr_opt_high_fs = pr_AIO;
da_AIO.opt_high_fs = OPT_REC;

pause(5);

close(BAR2);
close all;
clc;

%% collect / record data Low Fs
% the final performance test (1)

fprintf('\n\n??? Final Measurements Low Fs ... \n\n');
waitbar(7/TOTAL_ITEM, BAR, sprintf('All in One Test (7 / %d): Final Measurements Low Fs',TOTAL_ITEM));
dist_win;

i_bar2 = 0;
total_bar2 = 17;
BAR2 = waitbar(0,'');
BAR2.Position = [0,120,300,50];

pr_AIO = da_AIO.pr_opt_low_fs;

pr_AIO.N_run =          16;
pr_AIO.N_fft =          2^16;

% <the followings should be based on your test plan>

AA_test_single_tone;                % normal single-tone test
pr_AIO.single_end =     1;
AA_test_single_tone;                % single-ended test
pr_AIO.single_end =     0;
pr_AIO.refStb = 1-pr_AIO.refStb;    
AA_test_single_tone;                % turn-on/off a configuration
pr_AIO.refStb = 1-pr_AIO.refStb;

pr_AIO.F_in_want =      2.3 * 10^3;
AA_test_single_tone;                % test a few different input frequencies
pr_AIO.single_end =     1;
AA_test_single_tone;
pr_AIO.single_end =     0;

pr_AIO.F_in_want =      4.9 * 10^3;
AA_test_single_tone;
pr_AIO.single_end =     1;
AA_test_single_tone;
pr_AIO.single_end =     0;

pr_AIO.F_in_want =      9.6 * 10^3;
AA_test_single_tone;
pr_AIO.single_end =     1;
AA_test_single_tone;
pr_AIO.single_end =     0;

pr_AIO.F_in_want =      0.9 * 10^3;

pr_AIO.F_in_want1 =     0.9 * 10^3;
pr_AIO.F_in_want2 =     1.1 * 10^3;
AA_test_two_tone;                   % two tone test

pr_AIO.F_in_want1 =     4.9 * 10^3;
pr_AIO.F_in_want2 =     5.1 * 10^3;
AA_test_two_tone;

pr_AIO.F_in_want1 =     9.6 * 10^3;
pr_AIO.F_in_want2 =     9.8 * 10^3;
AA_test_two_tone;

pr_AIO.N_run = 1;

pr_AIO.sweep_ref = 'F_in_want';
pr_AIO.sweep_list = [0.1,0.5:0.5:9.5,9.9]*10^3;
AA_parameter_sweeper;               % input frequency sweeping
pr_AIO.single_end =     1;
AA_parameter_sweeper;
pr_AIO.single_end =     0;

pr_AIO.sweep_ref = 'F_s';
pr_AIO.sweep_list = [1:20]*10^3;
pr_AIO.('V_in_comp') = 0;
AA_parameter_sweeper;               % sampling frequency sweeping
pr_AIO.V_in_comp = 1;

pr_AIO.sweep_ref = 'V_in_peak_uc';
pr_AIO.sweep_list = [0,10.^(-6:0.5:-0.5)*1,0.9:0.025:1.15];
AA_parameter_sweeper;               % dynamic range sweeping

pr_AIO.sweep_ref = 'AVDD_scalling';
pr_AIO.sweep_list = [0.9:0.025:1.1];
AA_parameter_sweeper;               % supply voltage sweeping

pause(5);

close(BAR2);
close all;
clc;

%% collect / record data High Fs
% the final performance test (2), similar to (1)

fprintf('\n\n??? Final Measurements High Fs ... \n\n');
waitbar(8/TOTAL_ITEM, BAR, sprintf('All in One Test (8 / %d): Final Measurements High Fs',TOTAL_ITEM));
dist_win;

i_bar2 = 0;
total_bar2 = 17;
BAR2 = waitbar(0,'');
BAR2.Position = [0,120,300,50];

pr_AIO = da_AIO.pr_opt_high_fs;

pr_AIO.N_run =          16;
pr_AIO.N_fft =          2^16;

pr_AIO.F_in_want =      0.9 * 10^3;

AA_test_single_tone;
pr_AIO.single_end =     1;
AA_test_single_tone;
pr_AIO.single_end =     0;

pr_AIO.F_in_want =      5.9 * 10^3;
AA_test_single_tone;
pr_AIO.single_end =     1;
AA_test_single_tone;
pr_AIO.single_end =     0;
pr_AIO.refStb = 1-pr_AIO.refStb;
AA_test_single_tone;
pr_AIO.refStb = 1-pr_AIO.refStb;

pr_AIO.F_in_want =      12.3 * 10^3;
AA_test_single_tone;
pr_AIO.single_end =     1;
AA_test_single_tone;
pr_AIO.single_end =     0;

pr_AIO.F_in_want =      24.7 * 10^3;
AA_test_single_tone;
pr_AIO.single_end =     1;
AA_test_single_tone;
pr_AIO.single_end =     0;

pr_AIO.F_in_want =      0.9 * 10^3;

pr_AIO.F_in_want1 =     0.9 * 10^3;
pr_AIO.F_in_want2 =     1.1 * 10^3;
AA_test_two_tone;

pr_AIO.F_in_want1 =     5.9 * 10^3;
pr_AIO.F_in_want2 =     6.1 * 10^3;
AA_test_two_tone;

pr_AIO.F_in_want1 =     12.4 * 10^3;
pr_AIO.F_in_want2 =     12.6 * 10^3;
AA_test_two_tone;

pr_AIO.F_in_want1 =     24.1 * 10^3;
pr_AIO.F_in_want2 =     24.3 * 10^3;
AA_test_two_tone;

pr_AIO.N_run = 1;

pr_AIO.sweep_ref = 'F_in_want';
pr_AIO.sweep_list = [0.1,0.5:0.5:24.5,24.9]*10^3;
AA_parameter_sweeper;
pr_AIO.single_end =     1;
AA_parameter_sweeper;
pr_AIO.single_end =     0;

pr_AIO.sweep_ref = 'F_s';
pr_AIO.sweep_list = [1:1:50]*10^3;
pr_AIO.V_in_comp = 0;
AA_parameter_sweeper;
pr_AIO.V_in_comp = 1;

pr_AIO.sweep_ref = 'V_in_peak_uc';
pr_AIO.sweep_list = [0,10.^(-6:0.5:-0.5)*1,0.9:0.025:1.15];
AA_parameter_sweeper;

pr_AIO.sweep_ref = 'AVDD_scalling';
pr_AIO.sweep_list = [0.9:0.025:1.1];
AA_parameter_sweeper;

pause(5);

close(BAR2);
close all;
clc;

%% End of test
% save all data and show the conclusive results

da_AIO.success = 1;

da_AIO.time_end = datetime;

save(sprintf('%s\\%s_chip%d_AllInOne_%s.mat', save_path, project_name, pr_AIO.chip_id, datetime('now','format','yyyyMMddHHmmss')),'pr_AIO','da_AIO','-v7.3');
fprintf('Data Recorded.\n\n');

fprintf('All in One Test Completed Successfully!\n\n');

fprintf('Shuting Down and exit...\n\n');

AA_shutdown;

disp(['Test starts at ',string(da_AIO.time_start)]);
disp(['Test ends at ',string(da_AIO.time_end)]);

hf102 = figure(102); hf102.Position = [30 538 560 420]; clf; 
plot(da_AIO.opt_noise_floor(:,7));
title('Noise Floor');
xlabel('Iteration');
ylabel('FoMs (dB)');

hf103 = figure(103); hf103.Position = [623 538 560 420]; clf;
plot(da_AIO.opt_low_fs(:,7));
title('Fs = 20K');
xlabel('Iteration');
ylabel('FoMs (dB)');

hf104 = figure(104); hf104.Position = [1226 538 560 420]; clf;
plot(da_AIO.opt_high_fs(:,7));
title('Fs = 50K');
xlabel('Iteration');
ylabel('FoMs (dB)');

clear pr_AIO
clear da_AIO

close(BAR);

