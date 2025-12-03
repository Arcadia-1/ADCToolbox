close all; clc; clear; warning("off")

%% Configuration
verbose = 1;
inputDir = "test_dataset";
outputDir = "test_output";
figureDir = "test_plots";

filesList ={};
filesList = autoSearchFiles(filesList, inputDir, 'dout_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end
