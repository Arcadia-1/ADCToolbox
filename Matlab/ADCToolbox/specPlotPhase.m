function [h] = specPlotPhase(data,varargin)

p = inputParser;
validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
addOptional(p, 'N_fft', length(data), validScalarPosNum);
addOptional(p, 'harmonic', 5);
addParameter(p, 'OSR', 1, validScalarPosNum);
parse(p, varargin{:});

%%
N_fft = p.Results.N_fft;
harmonic = p.Results.harmonic;
OSR = p.Results.OSR;

%%

[N,M] = size(data);
if(M==1 && N > 1)
    data = data';
end

[N_run,~] = size(data);

Nd2 = floor(N_fft/2/OSR);

spec = zeros([1,N_fft]);
ME = 0;
for iter = 1:N_run
    tdata = data(iter,:);
    if(rms(tdata)==0)
        continue;
    end
    tdata = tdata./(max(tdata)-min(tdata));
    tdata = tdata-mean(tdata);

    tspec = fft(tdata);
    tspec(1) = 0;
    [~, bin] = max(abs(tspec(1:floor(N_fft/2/OSR))));
    phi = tspec(bin)/abs(tspec(bin));

    phasor = conj(phi);
    marker = zeros(1,N_fft);
    for iter2 = 1:N_fft                         % harmonic phase shift
        J = (bin-1)*iter2;
        if(mod(floor(J/N_fft*2),2) == 0)
            b = J-floor(J/N_fft)*N_fft+1;
            if(marker(b) == 0)
                tspec(b) = tspec(b).*phasor;
                marker(b) = 1;
            end
        else
            b = N_fft-J+floor(J/N_fft)*N_fft+1;
            if(marker(b) == 0)
                tspec(b) = tspec(b).*conj(phasor);
                marker(b) = 1;
            end
        end
        phasor = phasor * conj(phi);
    end

    for iter2 = 1:N_fft                         % non-harmonic shift
        if(marker(iter) == 0)
            tspec(iter) = tspec(iter).*(conj(phi).^((iter-1)/(bin-1)));
        end
    end

    spec = spec + tspec;

    ME = ME+1;
end

spec = spec(1:Nd2);

phi = spec./abs(spec);
spec = 10*log10(abs(spec).^2/(N_fft^2)*16/ME^2);
spec_sort = sort(spec);
minR = spec_sort(ceil(length(spec_sort)*0.01));
if(isinf(minR))
    minR = -100;
end
spec = max(spec,minR) - minR;
[~, bin] = max(spec);

spec = phi.*spec;

h = polarplot(spec,'k.');
grid on;
hold on;
pax = gca;
pax.ThetaDir = 'clockwise';
pax.ThetaZeroLocation = 'top';
tick = [-minR:-10:0];
pax.RTick = tick(end:-1:1);
tickl = (0:-10:minR);
pax.RTickLabel = tickl(end:-1:1);
rlim([0,-minR]);

polarplot(spec(bin),'ro');
polarplot([0,spec(bin)],'r-','linewidth',2);
for iter = 2:harmonic
    b = alias((bin-1)*iter,N_fft);
    polarplot(spec(b+1),'bs');  
    polarplot([0,spec(b+1)],'b-','linewidth',2);
    text(angle(spec(b+1))+0.1,abs(spec(b+1)),num2str(iter),'fontname','Arial','fontsize',8,'horizontalalignment','center');
end


title('Spectrum Phase');
end