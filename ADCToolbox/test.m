
N = 1000;
Fin = rand*0.5;
t = 0:N-1;
phase = rand*2*pi;
x = cos(Fin*2*pi*t+phase);
% plot(x)
% specPlot(x)

% %%
% N = length(data_cal);
% [~,Fin] = sineFit(data_cal);
% x = data_cal';

%%
Nbin = max(round(alias(Fin,1)*N),1);
Fbase = Fin/Nbin;

theta_mat = (0:(N-1))'*Fbase*(1:N/2);
xc = cos(theta_mat*2*pi);
xs = sin(theta_mat*2*pi);
A = [ones(N,1),xc,xs];
s = linsolve(A,x');

spec = [abs(s(1));sqrt(s(2:N/2+1).^2+s(N/2+2:N+1).^2)];
freq = zeros(1,N/2+1);
for ii = 1:N/2+1
    freq(ii) = Fbase*(ii-1);
end

plot(freq,20*log10(spec));

%%

figure(1);
specPlot(x,'NC',1);

figure(2);
specPlot(x);
