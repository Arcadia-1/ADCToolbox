% this is for retriving a single tone testing result and make figures

close all;
clc;

% specify the data to be retrived
% use CC_data_brower.m to find the data time stamp
chip_id = 2;
year = '2021';
month = '08';
day = '26';
time = '180146';

path_setting;

file_name = ['\ZIC_chip',num2str(chip_id),'_SingleTone_',year,month,day,time,'.mat'];
fprintf('%s\n\n',file_name);
load([data_path,file_name]);

pr_AIO = pr;

data_cal = da.data_coarse*pr.weight_MSB + da.data_fine;
analysis_single_tone;   % perform single tone analysis


fprintf('\tANA\t\tAMP\t\tVREF1\t\tVREF2\t\tDIG1\t\tDIG2\t\tDIG3\t|\tCLK\t\tIOL\t\tIOH\t\tIREF1\t\tIREF2\n');
fprintf('V:\t%.2f\t%.2f\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t%.2f\t%.2f\n',da.VVana,da.VVamp,da.VVref1,da.VVref2,da.VVdig1,da.VVdig2,da.VVdig3,da.VVclk,da.VViol,da.VVioh);
fprintf('uA:\t%.0f\t\t%.0f\t\t%.0f\t\t\t%.0f\t\t\t%.0f\t\t\t%.0f\t\t\t%.0f\t\t\t%.0f\t\t%.0f\t\t%.0f\t\t%.2f\t\t%.2f\n',...
    da.IVana*10^6,da.IVamp*10^6,da.IVref1*10^6,da.IVref2*10^6,da.IVdig1*10^6,da.IVdig2*10^6,da.IVdig3*10^6,...
    da.IVclk*10^6,da.IViol*10^6,da.IVioh*10^6,-pr.Iref1,-pr.Iref2);
fprintf('uW:\t%.0f\t\t%.0f\t\t%.0f\t\t\t%.0f\t\t\t%.0f\t\t\t%.0f\t\t\t%.0f\t\t\t%.0f\t\t%.0f\t\t%.0f\n',...
    da.VVana*da.IVana*10^6,da.VVamp*da.IVamp*10^6,da.VVref1*da.IVref1*10^6,da.VVref2*da.IVref2*10^6,da.VVdig1*da.IVdig1*10^6,...
    da.VVdig2*da.IVdig2*10^6,da.VVdig3*da.IVdig3*10^6,da.VVclk*da.IVclk*10^6,da.VViol*da.IViol*10^6,da.VVioh*da.IVioh*10^6);
fprintf('Total uW: \t\t\t\t\t\t\t\t\t\t\t\t\t\t\t%.0f\t\t\t\t\t\t\t%.0f\n',da.Power*10^6,(da.VVclk*da.IVclk+da.VViol*da.IViol+da.VVioh*da.IVioh)*10^6);
fprintf('Temperature: %.1f decC\n\n',da.Tenv);
fprintf('\n\tSNDR\t\tENOB\tSNR\t\t\tSFDR\t\tTHD\t\t\t\tFOMs\t\t\tFOMw\t\t\tP/Fs\n');
fprintf('\t%.1fdB\t\t%.1f\t%.1fdB\t\t%.1fdB\t\t%.1fdB\t\t%.1fdB\t\t\t%.1ffJ\t\t%.1fpJ\n\n',da.SNDR,(da.SNDR-1.76)/6.02,da.SNR,da.SFDR,da.THD,da.FOMs,da.FOMw*10^15,da.Power/pr.F_s*10^12);