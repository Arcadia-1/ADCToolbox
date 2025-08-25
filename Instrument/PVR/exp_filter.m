Fs = 2.8*10^9;

Fc = 160*10^6;

% freq = 10.^(3:0.01:10);
freq = [0.001:0.001:1]*10^10;

[mag_tf1] = bode(tf([1],[1/(Fc*2*pi), 1]),freq*2*pi);
mag_tf1 = mag_tf1(:);

[mag_tf2] = sinc(freq/Fs)';

K = 3.5/8;
[mag_iir] = bode(tf([(1-K)],[1, -K],1/Fs)^4,freq*2*pi);
mag_iir = mag_iir(:);

[mag_fir] = bode(tf([1,1,1,1,1,1,1,1]/8,[1,0,0,0,0,0,0,0],1/Fs),freq*2*pi);
mag_fir = mag_fir(:);

figure(1);
clf;
plot(freq,20*log10(abs(mag_tf1)));
hold on;
plot(freq,20*log10(abs(mag_tf2)));
plot(freq,20*log10(abs(mag_iir)));
plot(freq,20*log10(abs(mag_fir)));

plot(freq,20*log10(abs(mag_tf2).*abs(mag_iir).*abs(mag_fir)),'k-');

legend('RC','Sinc','IIR','FIR','total');

for i1 = 1:5
    plot([Fc*i1,Fc*i1],[-100,0],'k--');
end

axis([0,10*10^9,-60,0]);