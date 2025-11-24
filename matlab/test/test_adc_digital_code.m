
close all; clc; clear; 

%% specPlot and specPlotPhase must be usable on single row/column or batch data
read_data = readmatrix("Sine_wave_13_69_bit.csv");
figure;
specPlot(read_data','label', 1, 'harmonic', 0, 'OSR',1, 'coAvg', 0);
saveas(gcf, "output/specPlot_of_Sine_wave_13_69_bit_csv.png");

read_data_set = readmatrix("Sine_wave_13_69_bit_16run.csv");
figure;
specPlot(read_data_set,'label', 1, 'harmonic', 0, 'OSR',1, 'coAvg', 0);
saveas(gcf, "output/specPlot_of_Sine_wave_13_69_bit_16run.png");


figure;
specPlotPhase(read_data, 'harmonic', 50);
saveas(gcf, "output/specPlotPhase_of_Sine_wave_13_69_bit.png");

figure;
specPlotPhase(read_data_set, 'harmonic', 50);
saveas(gcf, "output/specPlotPhase_of_Sine_wave_13_69_bit_16run.png");

%% tomDecomp only need to work on a single row/col
N = length(read_data);
[~,relative_freq,~,~,~] = sineFit(read_data);
J = findBin(1,relative_freq,N);

figure;
[signal, error, indep, dep] = tomDecomp(read_data, J/N, 50, 1);
saveas(gcf, "output/tomDecomp_of_Sine_wave_13_69_bit.png");

%%
read_data = readmatrix("Sine_wave_13_69_bit.csv");
N = length(read_data);
[~,relative_freq,~,~,~] = sineFit(read_data);
J = findBin(1,relative_freq,N);
figure;
[emean, erms, phase_code, edata] = ...
    errHistSine(read_data*2^13, 1000, J/N, 1, 0, 'erange', [20,210]);

saveas(gcf, "output/errHistSine_of_Sine_wave_13_69_bit.png");

figure;
hist(edata,50);
saveas(gcf, "output/hist_of_edata_of_Sine_wave_13_69_bit.png");