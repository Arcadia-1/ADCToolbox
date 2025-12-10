close all; clc; clear; warning("off")

%% Configuration
verbose = 1;
% inputDir = "reference_dataset";
% outputDir = "reference_output"; % will be commited to GitHub!!
inputDir = "test_dataset";
outputDir = "test_output";
figureDir = "test_plots";

filesList ={"sinewave_nonlin_HD2_n80dB_HD3_n73dB"};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end
