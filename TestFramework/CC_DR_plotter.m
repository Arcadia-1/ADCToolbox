% this is for dynamic range measurement after a DR sweeping (Ain sweeping)
% it automatically measures the peak and x intersection in the DR sweeping 

clf;
plot(da.amp_rec,da.SNDR_rec,'o-','linewidth',2);
hold on;
plot(da.amp_rec,da.SNR_rec,'s-','linewidth',2);
plot(da.amp_rec,da.SFDR_rec,'d-','linewidth',2);
ylabel('dB');
xlabel('Signal Magnitude (dBFS)');
grid on;
legend('SNDR','SNR','SFDR');

k = (da.SNDR_rec(3)-da.SNDR_rec(2))/(da.amp_rec(3)-da.amp_rec(2));
cross_i = da.amp_rec(2)-da.SNDR_rec(2)/k;
[peak_sndr,peak_i] = max(da.SNDR_rec);
DR = da.amp_rec(peak_i)-cross_i;

fprintf('DR = %.1dB\n\n', DR);