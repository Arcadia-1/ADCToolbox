
close all; clc; clear; 

%% specPlot and specPlotPhase must be usable on single row/column or batch data
read_data_set = readmatrix("reference_data/Sine_wave_13_69_bit_16run.csv");
figure;
specPlot(read_data_set,'label', 1, 'harmonic', 0, 'OSR',1, 'coAvg', 0);
saveas(gcf, "output/specPlot_of_Sine_wave_13_69_bit_16run.png");

figure;
specPlotPhase(read_data_set, 'harmonic', 50);
saveas(gcf, "output/specPlotPhase_of_Sine_wave_13_69_bit_16run.png");