[M,N] = size(data_cal);

SNDR_rec = [];
SFDR_rec = [];
SNR_rec = [];
THD_rec = [];

figure(1);

for i1 = 1:M
    clf;
    [ENOB,SNDR,SFDR,SNR,THD] = specPlot(data_cal(i1,:), pr.F_s, 33*2^14, -5, @blackman, 2^13, 0, 1, NaN);
    
    SNDR_rec = [SNDR_rec,SNDR];
    SFDR_rec = [SFDR_rec,SFDR];
    SNR_rec = [SNR_rec,SNR];
    THD_rec = [THD_rec,THD];
end

clf;
hold on;
stem(1:M,SNDR_rec);
stem(1:M,SFDR_rec);
stem(1:M,SNR_rec);
stem(1:M,THD_rec);

legend('SNDR','SFDR','SNR','THD');


figure(2);
[ENOB,SNDR,SFDR,SNR,THD] = specPlot(data_cal([1,2,5:12,14:58,60:64],:), pr.F_s, 33*2^14, -5, @blackman, 2^13, 0, 1, NaN);

figure(3);
calcLinearity(data_cal([1,2,5:12,14:58,60:64],:));