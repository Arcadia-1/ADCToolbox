% this is a useful tool for searching the testing database and pick your
% interested data. it is also helpful for viewing the performance distribution
% between devices

path_setting;

% <specify the range of search>
data_type = 3;                % 1: SingleTone, 2:TwoTone, 3:Sweep
sweep_type = 'cycGap';        % effective only when data_type = 3, empty to match all

% <specify the conditions of search>

id_condition = [];  % the chip id, empty vector matches all chip

% for variables in "pr", condition format: '<pr name>',{[<min>,<max>] | <val>}
% <min>,<max> specify the range of coresponding parameter, or using <val> to specify the exact parameter value
% "and logic" is applied between conditions

para_condition = {  
    'V_in_peak',{[0.8,2]}
%     'F_s',{3*10^3}
%     'F_in_want',{5.9*10^3}
%     'N_fft', {2^16}
%     'N_run', {4}
%     'single_end', {1}
%     'PN_freq',{[0,2*10^6]}
%     'chopSel',{0}
    };

% for variables in "da", same format as above
data_condition = { 
%     'Noise', {[90,120]},
%     'DC',{[-2*10^-6,2*10^-6]}
    };

% the condition for time stamp, empty matchs all
year_condition = [];        % year
month_condition = [];       % month
day_condition = [];         % day
hour_condition = [];        % hour

% <specify the parameter (x-axis) and performance (y-axis) for the final scatter plot
xais = 'cycGap';      % scatter plot
yais = 'SNDR';

%% Searching and Loading

if(data_path(end) ~= '\')
    data_path = [data_path,'\'];
end

data_type_list = {'SingleTone','TwoTone','Sweep'};  % change this according to your test plan

if(data_type == 3)
    if(isempty(sweep_type))
        data_type_name = [data_type_list{data_type},'_*'];
    else
        data_type_name = [data_type_list{data_type},'_',sweep_type];
    end
else
    data_type_name = data_type_list{data_type};
end
if(isempty(id_condition))
    pat_id = '\d*';
else
    pat_id = sprintf('%d|',id_condition); pat_id = pat_id(1:end-1);
end
if(isempty(year_condition))
    pat_y = '\d{4}';
else
    pat_y = sprintf('%04d|',year_condition); pat_y = pat_y(1:end-1);
end
if(isempty(month_condition))
    pat_m = '\d{2}';
else
    pat_m = sprintf('%02d|',month_condition); pat_m = pat_m(1:end-1);
end
if(isempty(day_condition))
    pat_d = '\d{2}';
else
    pat_d = sprintf('%02d|',day_condition); pat_d = pat_d(1:end-1);
end
if(isempty(hour_condition))
    pat_h = '\d{2}';
else
    pat_h = sprintf('%02d|',hour_condition); pat_h = pat_h(1:end-1);
end

file_list = dir([data_path, project_name, '_chip*_',data_type_name,'_*.mat']);
n_file = length(file_list);

clear data_list;

close all;

bar_search = waitbar(0,'');
for i_search = 1:n_file
    nm = file_list(i_search).name;
    
    bar_search = waitbar(i_search/n_file,bar_search,sprintf('Loading File (%d,%d):\n%s',i_search, n_file, nm));
    
    if(data_type == 3)
        if(isempty(sweep_type))
            data_type_name = [data_type_list{data_type},'_[A-Za-z]*'];
        else
            data_type_name = [data_type_list{data_type},'_',sweep_type];
        end
    else
        data_type_name = data_type_list{data_type};
    end
        
    if(isempty(regexp(nm,[project_name,'_chip(',pat_id,')_',data_type_name,'_(',pat_y,')(',pat_m,')(',pat_d,')(',pat_h,')(\d{4}).mat'],'ONCE')))
        continue;
    end
    
    MATCH = 1;
    
    try
        
    load([file_list(i_search).folder,'\',nm],'pr');
    [pcn,pcm] = size(para_condition);
    for i_cond = 1:pcn
        para_name = para_condition{i_cond,1};
        if(isfield(pr,para_name))
            para_req_list = para_condition{i_cond,2};
            for i_req = 1:length(para_req_list)
                para_req = para_req_list{i_req};
                if(length(para_req) == 1)
                    if(pr.(para_name) ~= para_req)
                        MATCH = 0;
                        break;
                    end
                elseif(length(para_req) == 2)
                    if(pr.(para_name) < para_req(1) || pr.(para_name) > para_req(2))
                        MATCH = 0;
                        break;
                    end
                end
            end
            if(MATCH == 0)
                break;
            end
        else
            MATCH = 0;
            break
        end
    end
    
    catch E
        E
        fprintf('Err1 with %s\n\n',nm);
        MATCH = 0;
        continue;
    end
    
    if(MATCH == 0)
        continue;
    end
 
% <this part is optional for fixing the data file with missing results>
    
%     try
%             
%     load([file_list(i_search).folder,'\',nm],'da');
% 
%     if(~isfield(da,'SNDR'))
%         if(data_type == 1 || data_type == 2)
%             da.sideBinDiv = 2^9;
%             data = da.data_coarse*pr.weight_MSB + da.data_fine;
%             if(data_type == 1)
%                 [ENOB,SNDR,SFDR,SNR,THD,Signal,Noise,~] = specPlot(data, pr.F_s, 33*2^(pr.osrSel+7), 21, @blackman, pr.N_fft/da.sideBinDiv, 0, 0, NaN, 0);
%                 da.Signal = Signal;
%             else
%                 [ENOB,SNDR,SFDR,SNR,THD,s1,s2,Noise,~] = specPlot2Tone(data, pr.F_s, 33*2^(pr.osrSel+7), 7, @blackman, pr.N_fft/da.sideBinDiv, 0);
%                 da.Signal = 10*log10(10^(s1/10)+10^(s2/10));
%             end
%             da.ENOB = ENOB;
%             da.SNDR = SNDR;
%             da.SFDR = SFDR;
%             da.SNR = SNR;
%             da.THD = THD;
%             da.Noise = Noise;
%             da.Power = da.VVana*da.IVana+da.VVamp*da.IVamp+da.VVref1*da.IVref1+da.VVref2*da.IVref2+da.VVdig1*da.IVdig1+da.VVdig2*da.IVdig2+da.VVdig3*da.IVdig3;
%             da.DC = mean(data) / (33*2^(pr.osrSel+7)) * 2 - 1;
%             da.FOMs = SNDR+10*log10(pr.F_s/2/da.Power);
%             da.FOMw = da.Power/pr.F_s/2^((SNDR-1.76)/6.02);
%         elseif(data_type == 3)
%             [FOMs,i_best] = max(da.FOM_rec);
%             da.SNDR = da.SNDR_rec(i_best);
%             da.SNR = da.SNR_rec(i_best);
%             da.SFDR = da.SFDR_rec(i_best);
%             da.THD = da.THD_rec(i_best);
%             da.Signal = da.amp_rec(i_best);
%             da.Noise = da.SNR_rec(i_best)-da.amp_rec(i_best);
%             da.Power = da.pwr_rec(i_best);
%             if(isfield(da,'DC_rec'))
%                 da.DC = da.DC_rec(i_best);
%             else
%                 da.DC = 0;
%             end
%             da.FOMs = FOMs;
%             da.FOMw = da.Power/pr.F_s/2^((da.SNDR-1.76)/6.02);
%         end
%         save([file_list(i_search).folder,'\',nm],'da','-append');
%         fprintf('Appended data to %s\n\n',nm);
%     end
%     
%     catch E
%         E
%         fprintf('Err2 with %s\n\n',nm);
%         MATCH = 0;
%         continue;
%     end
    
    [dcn,dcm] = size(data_condition);
    for i_cond = 1:dcn
        data_name = data_condition{i_cond,1};
        if(isfield(da,data_name))
            data_req_list = data_condition{i_cond,2};
            for i_req = 1:length(data_req_list)
                data_req = data_req_list{i_req};
                if(length(data_req) == 1)
                    if(da.(data_name) ~= data_req)
                        MATCH = 0;
                        break;
                    end
                elseif(length(data_req) == 2)
                    if(da.(data_name) < data_req(1) || da.(data_name) > data_req(2))
                        MATCH = 0;
                        break;
                    end
                end
            end
            if(MATCH == 0)
                break;
            end
        else
            MATCH = 0;
            break
        end
    end
    
    if(MATCH == 0)
        continue;
    end
    
    try
        
    s = struct(...
        'name',nm,...
        'id',pr.chip_id,...
        'Fs',pr.F_s,...
        'Fin',pr.F_in_want,...
        'SNDR',da.SNDR,...
        'SNR',da.SNR,...
        'SFDR',da.SFDR,...
        'THD',da.THD,...
        'Signal',da.Signal,...
        'Noise',da.Noise,...
        'Power',da.Power,...
        'DC',da.DC,...
        'FOMs',da.FOMs,...
        'FOMw',da.FOMw,...
        'Temp',da.Tenv...
    );
    if(isfield(pr,xais))
        s.(xais) = pr.(xais);
    end
    if(isfield(pr,yais))
        s.(yais) = pr.(yais);
    end
    if(isfield(da,xais))
        s.(xais) = da.(xais);
    end
    if(isfield(da,yais))
        s.(yais) = da.(yais);
    end

    if(exist('data_list','var'))
        data_list(end+1) = s;
    else
        data_list = s;
    end

    catch E
        E
        fprintf('Err3 with %s\n\n',nm);
    end
    
end
close(bar_search)

%% Data Plot
labelSet = {
    'ks','ko','kd','k*','k+','kx',...
    'rs','ro','rd','r*','r+','rx',...
    'bs','bo','bd','b*','b+','bx',...
    'ys','yo','yd','y*','y+','yx',...
    'gs','go','gd','g*','g+','gx'
    };

figure(101);
clf;
subplot(1,2,1);
hold on;
for i_plot = 1:length(data_list)
    hp = plot(data_list(i_plot).id,data_list(i_plot).SNDR,labelSet{data_list(i_plot).id},'markersize',6,'linewidth',1);
    hp.DataTipTemplate.DataTipRows(1) = dataTipTextRow('File',{data_list(i_plot).name});
    hp.DataTipTemplate.DataTipRows(2) = dataTipTextRow('Chip',{num2str(data_list(i_plot).id)});
    hp.DataTipTemplate.DataTipRows(3) = dataTipTextRow('SNDR',{num2str(data_list(i_plot).SNDR,'%.1f dB')});
    hp.DataTipTemplate.DataTipRows(4) = dataTipTextRow('Fs',{num2str(data_list(i_plot).Fs/1000,'%.1f KHz')});
    hp.DataTipTemplate.DataTipRows(5) = dataTipTextRow('Fin',{num2str(data_list(i_plot).Fin/1000,'%.1f KHz')});
    hp.DataTipTemplate.DataTipRows(6) = dataTipTextRow('FOMs',{num2str(data_list(i_plot).FOMs,'%.1f dB')});
end
grid on;
xlabel('Chip ID');
ylabel('SNDR (dB)');

subplot(1,2,2);
hold on;
for i_plot = 1:length(data_list)
    hp = plot(data_list(i_plot).id,data_list(i_plot).FOMs,labelSet{data_list(i_plot).id},'markersize',6,'linewidth',1);
    hp.DataTipTemplate.DataTipRows(1) = dataTipTextRow('File',{data_list(i_plot).name});
    hp.DataTipTemplate.DataTipRows(2) = dataTipTextRow('Chip',{num2str(data_list(i_plot).id)});
    hp.DataTipTemplate.DataTipRows(3) = dataTipTextRow('SNDR',{num2str(data_list(i_plot).SNDR,'%.1f dB')});
    hp.DataTipTemplate.DataTipRows(4) = dataTipTextRow('Fs',{num2str(data_list(i_plot).Fs/1000,'%.1f KHz')});
    hp.DataTipTemplate.DataTipRows(5) = dataTipTextRow('Fin',{num2str(data_list(i_plot).Fin/1000,'%.1f KHz')});
    hp.DataTipTemplate.DataTipRows(6) = dataTipTextRow('FOMs',{num2str(data_list(i_plot).FOMs,'%.1f dB')});
end
grid on;
xlabel('Chip ID');
ylabel('FOMs (dB)');
dcm = datacursormode;
dcm.Enable = 'on';
dcm.DisplayStyle = 'window';

figure(102);
clf;
hold on;
for i_plot = 1:length(data_list)
    hp = plot(data_list(i_plot).(xais),data_list(i_plot).(yais),labelSet{data_list(i_plot).id},'markersize',6,'linewidth',1);
    hp.DataTipTemplate.DataTipRows(1).Label = xais;
    hp.DataTipTemplate.DataTipRows(2).Label = yais;
    hp.DataTipTemplate.DataTipRows(3) = dataTipTextRow('File',{data_list(i_plot).name});
    hp.DataTipTemplate.DataTipRows(4) = dataTipTextRow('Chip',{num2str(data_list(i_plot).id)});
    hp.DataTipTemplate.DataTipRows(5) = dataTipTextRow('SNDR',{num2str(data_list(i_plot).SNDR,'%.1f dB')});
    hp.DataTipTemplate.DataTipRows(6) = dataTipTextRow('Fs',{num2str(data_list(i_plot).Fs/1000,'%.1f KHz')});
    hp.DataTipTemplate.DataTipRows(7) = dataTipTextRow('Fin',{num2str(data_list(i_plot).Fin/1000,'%.1f KHz')});
    hp.DataTipTemplate.DataTipRows(8) = dataTipTextRow('FOMs',{num2str(data_list(i_plot).FOMs,'%.1f dB')});
end
grid on;
xlabel(xais);
ylabel(yais);
dcm = datacursormode;
dcm.Enable = 'on';
dcm.DisplayStyle = 'window';
