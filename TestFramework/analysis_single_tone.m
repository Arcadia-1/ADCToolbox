% this part is completely related to your actual chip and testing plan
% some useful functions for ADC analysis can be found at https://cloud.tsinghua.edu.cn/d/aaccdc94e87f4c449af7/

N_plot = 1000;
assumedSignal = NaN;
sideBinDiv = 2^8;

if(exist('pr_AIO','var'))
    if(isfield(pr_AIO,'assumedSignal'))
        assumedSignal = pr_AIO.assumedSignal;
    end
end

figure(1);
clf;
plot(1:N_plot,data_cal(:,1:N_plot),'linewidth',1);

figure(2);
clf;
[ENOB,SNDR,SFDR,SNR,THD,amp,NF,~] = specPlot(data_cal, pr.F_s, 33*2^(pr.osrSel+7), 13, @blackman, pr.N_fft/sideBinDiv, 0, 1, assumedSignal);

if(~exist('da_AIO') && sum(sum(abs(data_cal-mean(data_cal)))) > 0)
figure(3);
    clf;
    [code,DNL,INL] = calcLinearity2(data_cal);
    subplot(2,1,1);
    plot(code/(33*2^(pr.osrSel+7)), DNL/(33*2^(pr.osrSel+7))*10^6);
    xlabel('Input/Vref');
    ylabel('ppm');
    title('DNL');
    subplot(2,1,2);
    plot(code/(33*2^(pr.osrSel+7)), INL/(33*2^(pr.osrSel+7))*10^6);
    xlabel('Input/Vref');
    ylabel('ppm');
    title('INL');
    hold on;
end