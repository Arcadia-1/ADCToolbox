close all; clc; clear; warning("off")

%% Configuration
verbose = 0;
inputDir = "reference_dataset";
outputDir = "test_output";
figureDir = "test_plots";

filesList ={};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end
