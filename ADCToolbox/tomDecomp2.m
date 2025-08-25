function [ser, noi] = fseries(data, re_fin, order)
    
    % Thompson Decomposition
    % data - ADC data to be decomposed, 1xN vector
    % re_fin - relative input frequency, i.e., re_fin = f_in/f_sample
    % order - order of harmonics to be counted as dependent error, order = 1 counts the fundamental only

    S = size(data);
    if(S(1) < S(2))
        data = data';
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
    
end