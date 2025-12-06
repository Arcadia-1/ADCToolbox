close all; clc; clear; warning("off")

%% Configuration
verbose = 1;
inputDir = "test_dataset";
outputDir = "test_output";
figureDir = "test_plots";

filesList ={"sinewave_noise_0uV"};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end
