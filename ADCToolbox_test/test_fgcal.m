clear;
close all;
clc;

file_list = [; ...
    "digital_code_SAR_10_bit.csv"; ...
    "digital_code_SAR_11_bit.csv"; ...
    "digital_code_SAR_12_bit.csv"; ...
    ];

output_dir = "output";
if ~exist(output_dir, "dir")
    mkdir(output_dir);
end

for k = 1:length(file_list)

    filename = file_list(k);
    name_without_ext = erase(filename, ".csv");

    read_code = readmatrix("reference_data/"+filename);
    weight_cal = FGCalSine(read_code);
    figure;
    specPlot(read_code*weight_cal')

    saveas(gcf, output_dir+"/specPlot_of_"+name_without_ext+"_matlab.png");

end
