% this is for retriving a sweeping result and make figures

close all;
clc;

% specify the data to be retrived
% use CC_data_brower.m to find the data time stamp
chip_id = 6;
sweep_type = 'F_in_want';
year = '2021';
month = '08';
day = '27';
time = '181308';


RE_CALC = 1;    % re-analysis raw data for each sweeping point, this would be slower but it's useful if your analyzer updated
                % otherwize, use the recorded analysis results


%%

path_setting;

file_name = ['\ZIC_chip',num2str(chip_id),'_Sweep_',sweep_type,'_',year,month,day,time,'.mat'];
fprintf('%s\n\n',file_name);
load([data_path,file_name]);

pr_AIO = pr;

if(RE_CALC)
    for i_sweep = 1:length(pr.sweep_list)
        data_cal = reshape(da.data_coarse(i_sweep,:,:) * pr.weight_MSB + da.data_fine(i_sweep,:,:),[1,pr.N_fft*pr.N_run]);

        analysis_single_tone;
        da.SNDR_rec(i_sweep) = SNDR;
        da.SNR_rec(i_sweep) = SNR;
        da.SFDR_rec(i_sweep) = SFDR;
        da.THD_rec(i_sweep) = THD;
        da.amp_rec(i_sweep) = amp;
        da.pwr_rec(i_sweep) = da.VVana*da.IVana+da.VVamp*da.IVamp+da.VVref1*da.IVref1+da.VVref2*da.IVref2+da.VVdig1*da.IVdig1+da.VVdig2*da.IVdig2+da.VVdig3*da.IVdig3;
        da.FOM_rec(i_sweep) = SNDR+10*log10(pr.F_s/2/da.pwr_rec(i_sweep));
        da.DC_rec(i_sweep) = mean(data_cal) / (33*2^(pr.osrSel+7)) * 2 - 1;
    end
    save([data_path,file_name],'da','-append');     % write back the new analysis results to file
    fprintf('Appended data to %s\n\n',file_name);
end

% <plot the sweeping curve for interested specs>
figure(1);
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

figure(2);
plot(pr.sweep_list,da.amp_rec,'linewidth',2);
ylabel('dB');
xlabel(pr.sweep_ref);
grid on;
legend('AMP');

figure(3);
plot(pr.sweep_list,da.pwr_rec*10^6,'linewidth',2);
grid on;
xlabel(pr.sweep_ref);
ylabel('uW');
legend('Power');

figure(4);
plot(pr.sweep_list,da.FOM_rec,'linewidth',2);
grid on;
xlabel(pr.sweep_ref);
ylabel('dB');
legend('FoMs');

figure(5);
plot(pr.sweep_list,da.DC_rec,'linewidth',2);
grid on;
xlabel(pr.sweep_ref);
ylabel('Full Scale');
legend('Normalized DC');


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