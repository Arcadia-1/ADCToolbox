function [ENoB,SNDR,SFDR,SNR,THD,pwr1,pwr2,NF,IMD2,IMD3,h] = specPlot2Tone(data,Fs,maxCode,harmonic,winType,sideBin,label,isPlot)

    % update 22/05/22

    if(nargin < 2)
        Fs = 1;
    end
    if(nargin < 3)
        maxCode = max(data)-min(data);
    end
    if(nargin < 4)
        harmonic = 7;
    end
    if(nargin < 5)
        winType = @hann;
    end
    if(nargin < 6)
        sideBin = 1;
    end
    if(nargin < 7)
        label = 1;
    end
    if(nargin < 8)
        isPlot = 1;
    end

    [M,N] = size(data);
    
    Nd2 = floor(N/2);
    freq = (0:(Nd2-1))/N*Fs;
    win = window(winType,N,'periodic')';
    
    spec = zeros([1,N]);
    ME = 0;
    for iter = 1:M
        tdata = data(iter,:);
        if(rms(tdata)==0)
            continue;
        end
        tdata = tdata./maxCode;
        tdata = tdata-mean(tdata);
        tdata = tdata.*win/sqrt(mean(win.^2));
        spec = spec+abs(fft(tdata)).^2;
        ME = ME+1;
    end
    spec = spec(1:Nd2);
    spec(1:sideBin) = 0;
    spec = spec/(N^2)*16/ME;
    
    mins = min(10*log10(spec+10^-20));
    
    [~, bin1] = max(spec);
    t_spec = spec;
    t_spec(bin1) = 0;
    [~, bin2] = max(t_spec);
    if(bin1 > bin2)
        t = bin2;
        bin2 = bin1;
        bin1 = t;
    end
    
    sig1 = sum(spec(max(bin1-sideBin,1):min(bin1+sideBin,Nd2)));
    sig2 = sum(spec(max(bin2-sideBin,1):min(bin2+sideBin,Nd2)));
    pwr1 = 10*log10(sig1);
    pwr2 = 10*log10(sig2);
    
    if(isPlot)
        h = plot(freq,10*log10(spec));
        axis([Fs/N,Fs/2,mins,0]);
        grid on;
        hold on;
        if(label)
            plot(freq(max(bin1-sideBin,1):min(bin1+sideBin,Nd2)),10*log10(spec(max(bin1-sideBin,1):min(bin1+sideBin,Nd2))),'r-','linewidth',0.5);
            plot(freq(max(bin2-sideBin,1):min(bin2+sideBin,Nd2)),10*log10(spec(max(bin2-sideBin,1):min(bin2+sideBin,Nd2))),'r-','linewidth',0.5);
        end   
        
        if(harmonic > 0)
            for i = 2:harmonic
                for jj = 0:i
        
                    b = alias((bin1-1)*jj+(bin2-1)*(i-jj),N);
                    text(b/N*Fs,10*log10(spec(b+1)+10^(-20))+5,num2str(i),'fontname','Arial','fontsize',12,'horizontalalignment','center');
                    b = max(b-2,1):min(b+2,Nd2);
                    plot(b/N*Fs,10*log10(spec(b+1)+10^(-20)),'r-');

                    if(-(bin1-1)*jj+(bin2-1)*(i-jj)>0)
                        b = alias(-(bin1-1)*jj+(bin2-1)*(i-jj),N);
                        text(b/N*Fs,10*log10(spec(b+1)+10^(-20))+5,num2str(i),'fontname','Arial','fontsize',12,'horizontalalignment','center');
                        b = max(b-2,1):min(b+2,Nd2);
                        plot(b/N*Fs,10*log10(spec(b+1)+10^(-20)),'r-');
                    end

                    if((bin1-1)*jj-(bin2-1)*(i-jj)>0)
                        b = alias((bin1-1)*jj-(bin2-1)*(i-jj),N);
                        text(b/N*Fs,10*log10(spec(b+1)+10^(-20))+5,num2str(i),'fontname','Arial','fontsize',12,'horizontalalignment','center');
                        b = max(b-2,1):min(b+2,Nd2);
                        plot(b/N*Fs,10*log10(spec(b+1)+10^(-20)),'r-');
                    end
                end
            end
        end 
    else
        h = [];
    end
    
    spec(max(bin1-sideBin,1):min(bin1+sideBin,floor(N/2))) = 0;
    spec(max(bin2-sideBin,1):min(bin2+sideBin,floor(N/2))) = 0;
    
    noi = sum(spec);
    
    [~, sbin] = max(spec);
    spur = sum(spec(max(sbin-sideBin,1):min(sbin+sideBin,Nd2)));
    
    SNDR = 10*log10((sig1+sig2)/noi);
    SFDR = 10*log10((sig1+sig2)/spur);
    ENoB = (SNDR-1.76)/6.02;
    
    b = alias((bin1-1)+(bin2-1),N);
    spur21 = sum(spec(max(b+1-2,1):min(b+1+2,Nd2)));
    b = alias((bin2-1)-(bin1-1),N);
    spur22 = sum(spec(max(b+1-2,1):min(b+1+2,Nd2)));
    
    b = alias((bin1-1)*2+(bin2-1),N);
    spur31 = sum(spec(max(b+1-2,1):min(b+1+2,Nd2)));
    b = alias((bin1-1)+(bin2-1)*2,N);
    spur32 = sum(spec(max(b+1-2,1):min(b+1+2,Nd2)));
    b = alias((bin1-1)*2-(bin2-1),N);
    spur33 = sum(spec(max(b+1-2,1):min(b+1+2,Nd2)));
    b = alias((bin2-1)*2-(bin1-1),N);
    spur34 = sum(spec(max(b+1-2,1):min(b+1+2,Nd2)));
        
    IMD2 = 10*log10((sig1+sig2)/(spur21+spur22));
    IMD3 = 10*log10((sig1+sig2)/(spur31+spur32+spur33+spur34));
    
    thd = 0;
    for i = 2:floor(N/100)
        b = alias(bin2+(bin2-bin1)*(i-1)-1,N);
        thd = thd + sum(spec(max(b+1-2,1):min(b+1+2,Nd2)));
        spec(max(b+1-2,1):min(b+1+2,Nd2)) = 0;
        b = alias(bin1-(bin2-bin1)*(i-1)-1,N);
        thd = thd + sum(spec(max(b+1-2,1):min(b+1+2,Nd2)));
        spec(max(b+1-2,1):min(b+1+2,Nd2)) = 0;
    end
    noi = sum(spec);
    THD = 10*log10(thd/(sig1+sig2));
    SNR = 10*log10((sig1+sig2)/noi);
    NF = SNR-10*log10(sig1+sig2);
    
    if(isPlot)
        if(label)
            plot([1,1]*Fs/2,[0,-mins],'--');
            if(Fs > 10^6)
                text(Fs/N*2,mins*0.05,['Fs = ',num2str(Fs/10^6,'%.1f'),' MHz']);
                text((bin1-10)/N*Fs,pwr1-mins*0.05,[num2str(bin1/N*Fs/10^6,'%.1f MHz')],'HorizontalAlignment','right');
                text((bin1-10)/N*Fs,pwr1,[num2str(pwr1,'%.1f'), ' dB'],'HorizontalAlignment','right');
                text((bin2-10)/N*Fs,pwr2-mins*0.05,[num2str(bin2/N*Fs/10^6,'%.1f MHz')],'HorizontalAlignment','left');
                text((bin2-10)/N*Fs,pwr2,[num2str(pwr2,'%.1f'), ' dB'],'HorizontalAlignment','left');
            else
                text(Fs/N*2,mins*0.05,['Fs = ',num2str(Fs/10^3,'%.1f'),' kHz']);
                text((bin1-10)/N*Fs,pwr1-mins*0.05,[num2str(bin1/N*Fs/10^3,'%.1f kHz')],'HorizontalAlignment','right');
                text((bin1-10)/N*Fs,pwr1,[num2str(pwr1,'%.1f'), ' dB'],'HorizontalAlignment','right');
                text((bin2-10)/N*Fs,pwr2-mins*0.05,[num2str(bin2/N*Fs/10^3,'%.1f kHz')],'HorizontalAlignment','left');
                text((bin2-10)/N*Fs,pwr2,[num2str(pwr2,'%.1f'), ' dB'],'HorizontalAlignment','left');
            end

            text(Fs/N*2,mins*0.1,['ENoB = ',num2str(ENoB,'%.2f')]);
            text(Fs/N*2,mins*0.15,['SNDR = ',num2str(SNDR,'%.2f'),' dB']);
            text(Fs/N*2,mins*0.20,['SFDR = ',num2str(SFDR,'%.2f'),' dB']);
            text(Fs/N*2,mins*0.25,['SNR = ',num2str(SNR,'%.2f'),' dB']);
            text(Fs/N*2,mins*0.30,['Noise Floor = ',num2str(NF,'%.2f'),' dB']);
            text(Fs/N*2,mins*0.35,['IMD2 = ',num2str(IMD2,'%.2f'),' dB']);
            text(Fs/N*2,mins*0.40,['IMD3 = ',num2str(IMD3,'%.2f'),' dB']);
            
            xlabel('Freq (Hz)');
            ylabel('dBFS');
            title('Output Spectrum');
        end
    end
end