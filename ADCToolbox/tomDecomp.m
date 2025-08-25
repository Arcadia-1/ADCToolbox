function [signal, error, indep, dep, phi] = tomDecomp(data, re_fin, order, disp)
    
    % Thompson Decomposition
    % data - ADC data to be decomposed, 1xN vector
    % re_fin - relative input frequency, i.e., re_fin = f_in/f_sample
    % order - order of harmonics to be counted as dependent error, order = 1 counts the fundamental only
    % disp - turn on result display

    S = size(data);
    if(S(1) < S(2))
        data = data';
    end

    if(nargin < 2 || isnan(re_fin))
       re_fin = findFin(data);
    end
    if(nargin < 3)
        order = 10;
    end
    if(nargin < 4)
        disp = 1;
    end

    t = 0:(length(data)-1);
    
    SI = cos(t*re_fin*2*pi)';
    SQ = sin(t*re_fin*2*pi)';
    
    WI = mean(SI.*data)*2;
    WQ = mean(SQ.*data)*2;
    DC = mean(data);
    
    signal = DC + SI*WI + SQ*WQ;
    phi = -atan2(WQ,WI);
    
    SI = zeros([length(data),order]);
    SQ = zeros([length(data),order]);
    for ii = 1:order
      SI(:,ii) = cos(t*re_fin*ii*2*pi);
      SQ(:,ii) = sin(t*re_fin*ii*2*pi);
    end
    
    W = linsolve([SI,SQ],data);
    
    signal_all = DC + [SI,SQ] * W;
    
    error = data - signal;
    
    indep = data - signal_all; 
    
    dep = signal-signal_all;
    
    if(nargin >= 4 && disp)
        yyaxis left;
        plot(data,'kx');
        hold on;
        plot(signal,'-','color',[0.5,0.5,0.5]);
        axis([1,min(max(1.5/re_fin,100),length(data)), min(data)*1.1, max(data)*1.1]);
        ylabel('Signal');
        yyaxis right;
        plot(dep,'r-');
        hold on;
        plot(indep,'b-');
        axis([1,min(max(1.5/re_fin,100),length(data)), min(error)*1.1, max(error)*1.1]);
        ylabel('Error');
        xlabel('Samples');
        legend('data','signal','dependent err','independent err');
        
    end

end