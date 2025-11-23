close all; clear; clc;warning("off")

inputdir = "ADCToolbox_example_data";
outputdir = "ADCToolbox_example_output";

files_list = {
    'dout_SAR_12b_weight_1.csv';...
    'dout_SAR_12b_weight_2.csv'; ...
    'dout_SAR_12b_weight_3.csv'; ...
    };

if isempty(files_list)
    search_pattern = "dout*.csv";
    search_results = dir(fullfile(inputdir, search_pattern));
    files_list = {search_results.name};
end

% --- Processing Loop ---
for k = 1:length(files_list)
    current_filename = files_list{k};
    data_filepath = fullfile(inputdir, current_filename);
    read_code = readmatrix(data_filepath);
    weights_cal = FGCalSine(read_code);

    figure;
    overflowChk(read_code, weights_cal);

    [~, name, ~] = fileparts(current_filename);
    title_string = replace(name, '_', '\_');
    title(title_string)

    output_filepath = fullfile(outputdir, [name, '_overflowChk_matlab.png']);
    saveas(gcf, output_filepath);
    fprintf('[Saved image] -> [%s]\n\n', output_filepath);

end
