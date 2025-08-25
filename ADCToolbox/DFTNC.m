function [spec,freq] = DFTNC(x,fbase)

% DFT for non-coherent signals
% x - signal
% fbase - base frequency, i.e., the frequency of spec[1]

[M,N] = size(x);
if(M > 1)
    if(N > 1)
        error('x must be a vector');
    else
        x = x';
        [M,N] = size(x);
    end
end

if(nargin<2)
    fbase = 1/N;
end

Nd2 = floor(N/2);
theta_mat = (0:(N-1))'*fbase*(1:Nd2-1);
xc = cos(theta_mat*2*pi);
xs = sin(theta_mat*2*pi);
A = [ones(N,1),xc,xs];
spec = linsolve(A,x')';
spec = [spec(1),spec(2:Nd2)+spec(Nd2+1:2*Nd2-1)*1i];
freq = (0:Nd2-1)*fbase*2*pi;

end