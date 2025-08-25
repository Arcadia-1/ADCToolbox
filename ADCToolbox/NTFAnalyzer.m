function [noiSup] = NTFAnalyzer(NTF,Flow,Fhigh,isPlot)

% analyze the performance of NTF
% NTF - the noise transfer function (in z domain)
% Flow - low bound frequency of signal band (relative to Fs)
% Fhigh - high bound frequency of signal band (relative to Fs)
% noiSup - integrated noise suppresion of NTF in signal band in dB (compared to NTF=1)

    w = linspace(0,0.5,2^16);
    [mag,~] = bode(NTF,w*2*pi);
    mag = reshape(mag,size(w));

    np = sum(mag(w>Flow & w<Fhigh).^2)/length(w);
    noiSup = -10*log10(np);

    if(nargin == 4 && isPlot==1)
        semilogx(w,20*log10(mag));
        hold on;
        if(Flow > 0)
            semilogx([Flow,Flow],20*log10([min(mag),max(mag)]),'k--');
        end
        semilogx([Fhigh,Fhigh],20*log10([min(mag),max(mag)]),'k--');
    end

end
    