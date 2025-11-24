close all; clear; clc;
inputdir = "ADCToolbox_example_data";
outputdir = "ADCToolbox_example_output";
% --- RESOLUTION LIST TO SCAN ---
Resolution_list = [12];
if ~exist(outputdir, 'dir')
    mkdir(outputdir);
end
manual_files_list = {}; 
% Example: manually setting the list for demonstration clarity
% manual_files_list = {"sinewave_gain_error_1P02.csv"}; 
if ~isempty(manual_files_list)
    file_list_struct = struct('name', manual_files_list);
else
    search_patterns = ["sinewave_HD_*.csv", "sinewave_gain_error_*.csv"];
    all_files = [];
    
    for i = 1:length(search_patterns)
        current_search = dir(fullfile(inputdir, search_patterns{i}));
        all_files = [all_files; current_search]; 
    end
    
    [~, unique_indices] = unique({all_files.name}, 'stable');
    file_list_struct = all_files(unique_indices);
end
% --- 2. FILE PROCESSING LOOP (Outer Loop) ---
for k = 1:length(file_list_struct)
    current_filename = file_list_struct(k).name;
    data_filepath = fullfile(inputdir, current_filename);
    
    [~, name, ~] = fileparts(current_filename);
    
    % Data Loading (once per file)
    data = readmatrix(data_filepath);
    
    % --- 3. RESOLUTION SCAN LOOP (Inner Loop) ---
    for res_idx = 1:length(Resolution_list)
        Resolution = Resolution_list(res_idx);
        scaled_data = data * 2^Resolution;
        
        data_min = min(scaled_data);
        data_max = max(scaled_data);
        expected_max = 2^Resolution;        
        [INL, DNL, code] = INLsine(scaled_data);
    
        % --- 4. Plotting (Required to calculate INL/DNL ranges for printing) ---
        
        % Calculate DNL/INL ranges (required for output string)
        max_inl = max(INL);
        min_inl = min(INL);
        max_dnl = max(DNL);
        min_dnl = min(DNL);
        
        % --- Start Plotting and Saving Block ---
        figure('Visible', 'off'); 
        
        % Top Subplot: INL
        subplot(2, 1, 1);
        scatter(code, INL, 8, 'filled'); 
        xlabel('Code'); % 横坐标: Code
        ylabel('INL (LSB)'); % 纵坐标: INL (LSB)
        grid on;
        title_str_inl = sprintf('INL = [%.2f, %+.2f] LSB', min_inl, max_inl);
        title(title_str_inl);
        
        ylim([min(min_inl, -1), max(max_inl, 1)]); 
        xlim([0, expected_max]); 
        
        % Bottom Subplot: DNL
        subplot(2, 1, 2);
        scatter(code, DNL, 8, 'filled'); 
        xlabel('Code'); % 横坐标: Code
        ylabel('DNL (LSB)'); % 纵坐标: DNL (LSB)
        grid on;
        title_str_dnl = sprintf('DNL = [%.2f, %.2f] LSB', min_dnl, max_dnl);
        title(title_str_dnl);
        
        ylim([min(min_dnl, -1), max(max_dnl, 1)]); 
        xlim([0, expected_max]);
        
        subdir_path = fullfile(outputdir, name);
        if ~exist(subdir_path, 'dir')
            mkdir(subdir_path);
        end
        output_filename_base = sprintf('INL_%db_%s_matlab', Resolution, name);
        output_filepath = fullfile(subdir_path, [output_filename_base, '.png']);
        
        saveas(gcf, output_filepath);
        close(gcf);
        % --- End Plotting and Saving Block ---

        % --- Final Console Output ---
        fprintf('[Resolution = %d]: DNL = [%.2f, %.2f] LSB, INL = [%.2f, %+.2f] LSB\n', ...
            Resolution, min_dnl, max_dnl, min_inl, max_inl);
        fprintf('[Saved image] -> [%s]\n\n', output_filepath);
    end
end