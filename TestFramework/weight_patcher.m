clc;
path_setting;

file_list = {
    '\ZIC_chip2_singleTone_20210826180146.mat',
    '\ZIC_chip6_singleTone_20210829144817.mat',
    '\ZIC_chip7_singleTone_20210829200107.mat',
    '\ZIC_chip9_singleTone_20210830133657.mat',
    '\ZIC_chip11_singleTone_20210830193934.mat',
    '\ZIC_chip13_singleTone_20210831182708.mat',
    '\ZIC_chip15_singleTone_20210831222721.mat',
    '\ZIC_chip16_singleTone_20210901022611.mat',
    '\ZIC_chip17_singleTone_20210901133301.mat',
    '\ZIC_chip18_singleTone_20210902033807.mat',
    '\ZIC_chip20_singleTone_20210902131930.mat',
    '\ZIC_chip21_singleTone_20210902174624.mat',
    '\ZIC_chip26_singleTone_20210903173612.mat',
    };

for i_file = 1:length(file_list)
    fprintf('(%d/%d) Patching %s ...\n',i_file,length(file_list),file_list{i_file});
    
    load([data_path,file_list{i_file}]);

    sweep_MSB_weight;
    
    [~,i_best] = max(SNDR_rec);
    pr.weight_MSB = gain_list(i_best);
    
    save([data_path,file_list{i_file}],'pr','-append');
    
    fprintf('Weight Updated! \n\n');
end
