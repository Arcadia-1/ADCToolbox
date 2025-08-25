% this part is completely related to your actual chip and testing plan
% some useful functions for ADC analysis can be found at https://cloud.tsinghua.edu.cn/d/aaccdc94e87f4c449af7/

N_plot = 1000;
sideBinDiv = 2^9;

figure(1);
clf;
plot(1:N_plot,data_final(:,1:N_plot),'linewidth',1);

% figure(2);
% clf;
% % specPlot(data,Fs,maxCode,harmonic,winType,sideBin,logSca,label,assumedSignal)
% specPlot(data_final, pr.F_s, 33*2^(pr.osrSel+7), 13, @blackman, pr.N_fft/pr.sideBinDiv, 0, 1, assumedSignal);

figure(2);
clf;
[ENOB,SNDR,SFDR,SNR,THD,amp,NF,~] = specPlot2Tone(data_cal, pr.F_s, 33*2^(pr.osrSel+7), 5, @blackman, pr.N_fft/sideBinDiv, 1);

% figure(3);
% clf;
% calcLinearity(data_cal);