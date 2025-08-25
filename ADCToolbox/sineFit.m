function [data_fit,freq,mag,dc,phi] = sineFit(data,f0,tol,rate)

    % data - the data to be fitted 
    % f0 - an estimated relative frequency (optional)
    % data_fit - the fitted sinewave

    [N,M] = size(data); 
    if(N == 1)
       data = data';
       N = M;
    end
    data = mean(data,2);
    
    if(nargin < 2)
        spec = abs(fft(data));
        spec(1) = 0;
        spec = spec(1:floor(N/2));
        
        [~,k0] = max(spec);
        if(spec(min(max(k0+1,1),N/2)) > spec(min(max(k0-1,1),N/2)))
            r = 1;
        else
            r = -1;
        end
        
        f0 = (k0-1 + r*spec(k0+r)/(spec(k0)+spec(k0+r)))/N;
        
    end

    if(nargin < 3)
        tol = 1e-12;
    end

    if(nargin < 4)
        rate = 0.5;
    end

    time = (0:N-1)';
    theta = 2*pi*f0*time;
    M = [cos(theta), sin(theta), ones([N,1])];
    x = linsolve(M,data);
    A = x(1);   % coefficient of cos
    B = x(2);   % coefficient of sin
    dc = x(3);   % DC component

    freq = f0;
    delta_f = 0;

    for ii = 1:100

        freq = freq+delta_f;
        theta = 2*pi*freq*time;
        M = [cos(theta), sin(theta), ones([N,1]), (-A*2*pi*time.*sin(theta)+B*2*pi*time.*cos(theta))/N];
        x = linsolve(M,data);
        A = x(1);
        B = x(2);
        dc = x(3);
        delta_f = x(4)*rate/N;
        relerr = rms(x(end)/N*A(:,end)) / sqrt(x(1)^2+x(2)^2);

        % fprintf('Freq fine iterating (%d): freq = %d, delta_f = %d, rel_err = %d\n', ii,freq,delta_f, relerr);
       
        if(relerr < tol)
            break;
        end

    end

    data_fit = (A*cos(theta)+B*sin(theta)+dc)';
    mag = sqrt(A^2+B^2);
    phi = -atan2(B,A);
end