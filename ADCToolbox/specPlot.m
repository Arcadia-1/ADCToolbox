function [ENoB,SNDR,SFDR,SNR,THD,pwr,NF,h] = specPlot(data,varargin)
 
p = inputParser;
validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
addOptional(p, 'Fs', 1, validScalarPosNum);
addOptional(p, 'maxCode', max(max(data))-min(min(data)), validScalarPosNum);
addOptional(p, 'harmonic', 5);
addParameter(p, 'OSR', 1, validScalarPosNum);
addParameter(p, 'winType', @hann);
addParameter(p, 'sideBin', 1, @(x) isnumeric(x) && isscalar(x) && (x >= 0));
addParameter(p, 'label', 1, @(x)ismember(x, [0, 1]));
addParameter(p, 'assumedSignal', NaN);
addParameter(p, 'isPlot', 1, @(x)ismember(x, [0, 1]));
addParameter(p, 'noFlicker', 0, validScalarPosNum);
addParameter(p, 'nTHD', 5, validScalarPosNum);
addParameter(p, 'coAvg', 0, @(x)ismember(x, [0, 1]));
addParameter(p, 'NFMethod', 0, @(x)ismember(x, [0, 1]))
parse(p, varargin{:});

%%
Fs = p.Results.Fs;
maxCode = p.Results.maxCode;
harmonic = p.Results.harmonic;
OSR = p.Results.OSR;
winType = p.Results.winType;
sideBin = p.Results.sideBin;
label = p.Results.label;
assumedSignal = p.Results.assumedSignal;
isPlot = p.Results.isPlot;
noFlicker = p.Results.noFlicker;
nTHD = p.Results.nTHD;
coAvg = p.Results.coAvg;
nfmethod = p.Results.NFMethod;

%%

[N,M] = size(data);
N_fft = M;
if(M==1 && N > 1)
    data = data';
    N_fft = N;
end

[N_run,~] = size(data);

Nd2 = floor(N_fft/2)+1;

freq = (0:(Nd2-1))/N_fft*Fs;

try
    win = window(winType,N_fft,'periodic')';
catch
    try
        win = window(winType,N_fft)';
    catch
        win = window(@rectwin,N_fft)';
        warning("Unsupported window type %s, using rectangle window", winType);
    end
end


spec = zeros([1,N_fft]);
ME = 0;
for iter = 1:N_run
    tdata = data(iter,:);
    if(rms(tdata)==0)
        continue;
    end
    tdata = tdata./maxCode;
    tdata = tdata-mean(tdata);
    tdata = tdata.*win/sqrt(mean(win.^2));

    if(coAvg)
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

    else
        spec = spec+abs(fft(tdata)).^2;
    end
    
    ME = ME+1;
end

if(coAvg)
    spec = abs(spec).^2/(N_fft^2)*16/ME^2;
else
    spec(1) = 0;
    spec = spec/(N_fft^2)*16/ME;
end
spec = spec(1:Nd2);
spec_inband = spec(1:floor(N_fft/2/OSR));


if noFlicker > 0 
    spec(1:ceil(noFlicker/Fs*N_fft)) = 0;
end


[~, bin] = max(spec_inband);
sig_e = log10(spec(bin));
sig_l = log10(spec(min(max(bin-1,1),Nd2)));
sig_r = log10(spec(min(max(bin+1,1),Nd2)));
bin_r = bin + (sig_r-sig_l)/(2*sig_e-sig_l-sig_r)/2;

sig = sum(spec(max(bin-sideBin,1):min(bin+sideBin,floor(N_fft/2/OSR))));
pwr = 10*log10(sig);
if(~isnan(assumedSignal))
    sig = 10.^(assumedSignal/10);
    pwr = assumedSignal;
end

if(harmonic < 0)
    for i = 2:-harmonic
        b = alias(round((bin_r-1)*i),N_fft);
        spec(max(b+1-sideBin,1):min(b+1+sideBin,Nd2)) = 0;
    end
end

if(isPlot)
    if (OSR == 1)
        h = plot(freq,10*log10(spec+10^(-20)));
    else
        h = semilogx(freq,10*log10(spec+10^(-20)));
    end

    grid on;
    hold on;
    if(label)
        if (OSR == 1)
            plot(freq(max(bin-sideBin,1):min(bin+sideBin,Nd2)),10*log10(spec(max(bin-sideBin,1):min(bin+sideBin,Nd2))),'r-','linewidth',0.5);
            plot(freq(bin),10*log10(spec(bin)),'ro','linewidth',0.5);
        else
            semilogx(freq(max(bin-sideBin,1):min(bin+sideBin,Nd2)),10*log10(spec(max(bin-sideBin,1):min(bin+sideBin,Nd2))),'r-','linewidth',0.5);
        end
    end
    if(harmonic > 0)
        for i = 2:harmonic
            b = alias(round((bin_r-1)*i),N_fft);
            plot(b/N_fft*Fs,10*log10(spec(b+1)+10^(-20)),'rs');
            text(b/N_fft*Fs,10*log10(spec(b+1)+10^(-20))+5,num2str(i),'fontname','Arial','fontsize',12,'horizontalalignment','center');
        end
    end
end

sigs = spec(bin);
if(~isnan(assumedSignal))
    sigs = 10.^(assumedSignal/10);
end
spec(max(bin-sideBin,1):min(bin+sideBin,Nd2)) = 0;
spec(1:sideBin) = 0;
spec_inband = spec(1:floor(N_fft/2/OSR));
noi = sum(spec_inband);

[spur, sbin] = max(spec_inband);
SNDR = 10*log10(sig/noi);
SFDR = 10*log10(sigs/spur);
ENoB = (SNDR-1.76)/6.02;

if(isPlot && label)
    plot((sbin-1)/N_fft*Fs,10*log10(spur+10^(-20)),'rd');
    text((sbin-1)/N_fft*Fs,10*log10(spur+10^(-20))+5,'MaxSpur','fontname','Arial','fontsize',10,'horizontalalignment','center');
end

thd = 0;
for i = 2:nTHD
    b = alias(round((bin_r-1)*i),N_fft) +1;
    thd = thd + spec(b);
    spec(b) = 0;
end

if(nfmethod == 0)
    noi = median(spec(1:floor(N_fft/2/OSR)))*floor(N_fft/2/OSR);
else 
    noi = sum(spec(1:floor(N_fft/2/OSR)));
end

THD = 10*log10(thd/sigs);
SNR = 10*log10(sig/noi);
NF = SNR - pwr;

if(isPlot)
    minx = min(max(median(10*log10(spec_inband))-20,-200),-40);
    axis([Fs/N_fft,Fs/2,minx,0]);
    if(label)
        plot([1,1]*Fs/2/OSR,[0,-1000],'--');
        if(OSR>1)
            TX = 10^(log10(Fs)*0.01+log10(Fs/N_fft)*0.99);
        else
            if(bin/N_fft < 0.2)
                TX = Fs*0.3;
            else
                TX = Fs*0.01;
            end
        end
        
        TYD = minx*0.06;

        if(Fs >= 10^9)
            txt_fs = num2str(Fs/10^9,'%.1fG');
        elseif(Fs >= 10^6)
            txt_fs = num2str(Fs/10^6,'%.1fM');
        elseif(Fs >= 10^3)
            txt_fs = num2str(Fs/10^3,'%.1fK');
        elseif(Fs >= 1)
            txt_fs = num2str(Fs,'%.1f');
        else
            txt_fs = num2str(Fs,'%.3f');
        end

        Fin = (bin_r-1)/N_fft*Fs;
        if(Fin >= 10^9)
            txt_fin = num2str(Fin/10^9,'%.1fG');
        elseif(Fin >= 10^6)
            txt_fin = num2str(Fin/10^6,'%.1fM');
        elseif(Fin >= 10^3)
            txt_fin = num2str(Fin/10^3,'%.1fK');
        elseif(Fin >= 1)
            txt_fin = num2str(Fin/10^3,'%.1f');
        else
            txt_fin = num2str(bin/N_fft*Fs,'%.3f');
        end

        text(TX,TYD,['Fin/Fs = ',txt_fin,' / ',txt_fs,' Hz']);
        
        text(TX,TYD*2,['ENoB = ',num2str(ENoB,'%.2f')]);
        text(TX,TYD*3,['SNDR = ',num2str(SNDR,'%.2f'),' dB']);
        text(TX,TYD*4,['SFDR = ',num2str(SFDR,'%.2f'),' dB']);
        text(TX,TYD*5,['THD = ',num2str(THD,'%.2f'),' dB']);
        text(TX,TYD*6,['SNR = ',num2str(SNR,'%.2f'),' dB']);
        text(TX,TYD*7,['Noise Floor = ',num2str(NF,'%.2f'),' dB']);

        text(bin/N_fft*Fs,pwr,['Sig = ',num2str(pwr,'%.2f'),' dB']);

        if (OSR>1)
            semilogx([Fs/N_fft,Fs/2/OSR],-[1,1]*(NF+10*log10(N_fft/2/OSR)),'r--');
            text(TX,TYD*8,['NSD = ',num2str(NF+10*log10(Fs/2/OSR),'%.2f'),' dBFS/Hz']);
            text(TX,TYD*9,['OSR = ',num2str(OSR,'%.2f')]);
        else
            plot([0,Fs/2],-[1,1]*(NF+10*log10(N_fft/2/OSR)),'r--');
            text(TX,TYD*8,['NSD = ',num2str(NF+10*log10(Fs/2/OSR),'%.2f'),' dBFS/Hz']);
        end
    end
    xlabel('Freq (Hz)');
    ylabel('dBFS');
    title('Output Spectrum');
end

if(~isPlot)
    h = [];
end
end