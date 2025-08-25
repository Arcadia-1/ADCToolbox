% find optimal interstage gain by linear search

SFDR_rec = [];
SNDR_rec = [];
gain_list = 31.5:0.0025:32; % this is design related

for gain = gain_list
    
    data_cal = da.data_coarse*gain + da.data_fine;
    
    figure(3);
    clf;
    [ENOB,SNDR,SFDR,SNR,THD,~] = specPlot(data_cal, pr.F_s, 33*2^14, 3, @blackman, 5, 0, 1, -1);
    
%     figure(4);
%     clf;
%     calcLinearity(data_cal,0.01);
    
    SFDR_rec = [SFDR_rec,SFDR];
    SNDR_rec = [SNDR_rec,SNDR];
    
    pause(0.01);
    
end

figure(5);
plot(gain_list,SFDR_rec);
hold on;
plot(gain_list,SNDR_rec);