run(fullfile(fileparts(mfilename('fullpath')), '..', 'utils', 'ensureMatlabRoot.m'));
close all; clc; clear; warning("off")

%% Configuration
verbose = 1;
% inputDir = "reference_dataset";
% outputDir = "reference_output"; % will be commited to GitHub!!
inputDir = "test_dataset";
if ~isfolder(inputDir) && isfolder(fullfile("..", "reference_dataset"))
    inputDir = fullfile("..", "reference_dataset");
end
outputDir = "test_output";
figureDir = "test_plots";

filesList ={};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end
