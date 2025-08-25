function fin = findFin(data,Fs)

if(nargin < 2)
    Fs = 1;
end
[~,freq,~,~,~] = sineFit(data);
fin = freq*Fs;

end