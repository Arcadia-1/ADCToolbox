function [weight,offset,postCal,ideal,err,freqCal] = FGCalSine(bits,varargin)

% foreground calibration by inputting sinewave
% data format: MSB left - LSB right
% bits is the raw data, N row by M col, N is data points, M is bitwidth
% freq is the reletive test freq Fin/Fs
% update 25/5 - freq is optional. auto freq search is enabled when freq is default or fsearch = 1
%               rate is the adaptive rate of freq search. default value 0.5 works for most of the case
%               reltol is the relative tolerance for freq search, default is 1e-6 (120dB)
%               
%
% order is the order of distortion excluded (default or order = 1 to include all distortion)
%
% nomWeight is the nominal weight (no normalization required), this is only
% effective when rank is deficient. The ratio of deficient weights are determined by
% this parameter then.
%
% weight is normalized by MSB. i.e., weigth[MSB] = 1. MSB defined as the first non-zero weight from left
% offset is normalized by MSB too
% postCal is the signal after calibration
% ideal is the ideal sine signal 
% err is the residue errors after calibration (excluding distortion)
% freqCal is the fine tuned frequency from calibration

    [N,M] = size(bits);
    if(N < M)
        bits = bits';
        [N,M] = size(bits);
    end

    p = inputParser;
    addOptional(p, 'freq', 0, @(x) isnumeric(x) && isscalar(x) && (x >= 0));
    addOptional(p, 'rate', 0.5, @(x) isnumeric(x) && isscalar(x) && (x > 0) && (x < 1));
    addOptional(p, 'reltol', 1E-12, @(x) isnumeric(x) && isscalar(x) && (x > 0));
    addOptional(p, 'niter', 100, @(x) isnumeric(x) && isscalar(x) && (x > 0));
    addOptional(p, 'order', 1, @(x) isnumeric(x) && isscalar(x) && (x > 0));
    addOptional(p, 'fsearch', 0, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'nomWeight', 2.^(M-1:-1:0));
    parse(p, varargin{:});
    freq = p.Results.freq;
    order = max(round(p.Results.order),1);
    nomWeight = p.Results.nomWeight;
    rate = p.Results.rate;
    reltol = p.Results.reltol;
    niter = p.Results.niter;
    fsearch = p.Results.fsearch;
    
    L = [1:M];              % link from a column to its correlated column
    K = ones(1,M);          % weight ratio of a column to its correlated column

    if(rank([bits,ones(N,1)]) < M+1)
        warning('Rank deficiency detected. Try patching...');
        bits_patch = [];
        LR = [];             % reverse link from bits_patch to bits
        M2 = 0;
        for i1 = 1:M
            if(max(bits(:,i1))==min(bits(:,i1)))    % constant column
                L(i1) = 0;
                K(i1) = 0;
            elseif(rank([ones(N,1),bits_patch,bits(:,i1)]) > rank([ones(N,1),bits_patch]))  % column i1 of bits is non corr to the rest columns
                bits_patch = [bits_patch,bits(:,i1)];
                LR = [LR,i1];
                [~,M2] = size(bits_patch);
                L(i1) = M2;     % normal column: link to itself
            else
                flag = 0;
                for i2 = 1:M2
                    r1 = bits(:,i1)-mean(bits(:,i1));
                    r2 = bits_patch(:,i2) - mean(bits_patch(:,i2));
                    cor = mean(r1.*r2)/rms(r1)/rms(r2);
                    if(abs(abs(cor)-1) < 1E-3)    % correlated column found
                        L(i1) = i2;     % link column i1 to column i2
                        K(i1) = nomWeight(i1)/nomWeight(LR(i2));
                        bits_patch(:,i2) = bits_patch(:,i2) + bits(:,i1)*nomWeight(i1)/nomWeight(LR(i2));   % merge the correlated column
                        flag = 1;
                        break;
                    end
                end
                if(flag == 0)                    
                    L(i1) = 0;
                    K(i1) = 0;
                    warning('Patch warning: cannot find the correlated column for column %d. The resulting weight will be zero',i1);
                end
            end          
        end
        [~,M] = size(bits_patch);
        if(rank([ones(N,1),bits_patch]) < M+1)
            error('Patch failed: rank still deficient after patching. This may be fixed by changing nomWeight.')
        end
    else
        bits_patch = bits;
    end

    % pre-scaling to avoid numerical problem
    MAG = floor(log10(max(abs([max(bits_patch);min(bits_patch)]))));
    MAG(isinf(MAG)) = 0;
    bits_patch = bits_patch.*(ones(N,1)*10.^(-MAG));
    
    if(freq == 0)
        fsearch = 1;
        freq = [];
        for i1 = 1:min(M,5)
            fprintf('Freq coarse searching (%d/5):',i1);
            freq = [freq, findFin(bits_patch(:,1:i1)*nomWeight(1:i1)')];
            fprintf(' freq = %d\n',freq(end));
        end
        freq = median(freq);
    end
    
    theta_mat = (0:(N-1))'*freq*(1:order);
    xc = cos(theta_mat*2*pi);
    xs = sin(theta_mat*2*pi);
    A = [bits_patch(1:N,1:M),ones(N,1),xc(:,2:end),xs];
    b = -xc(:,1);
    x1 = linsolve(A,b);
    A = [bits_patch(1:N,1:M),ones(N,1),xs(:,2:end),xc];
    b = -xs(:,1);
    x2 = linsolve(A,b);
    if(rms(A*x1-b) < rms(A*x2-b))
        x = x1;
        sel = 0;
    else
        x = x2;
        sel = 1;
    end
    
    
    if(fsearch)

        warning off;

        delta_f = 0;
        time_mat = (0:(N-1))'*ones([1,order]);
        
        for ii = 1:niter
            freq = freq+delta_f;
            theta_mat = (0:(N-1))'*freq*(1:order);          
            
            xc = cos(theta_mat*2*pi);  % cosine and harmonic components
            xs = sin(theta_mat*2*pi);  % sine and harmonic components
            
            order_mat = ones([N,1]) * (1:order);
            if(sel)
                KS = ones([N,1]) * [1,x(M+2:M+order)'] .* order_mat;        % coefficient of sin
                KC = ones([N,1]) * x(M+1+order:M+order*2)' .* order_mat;    % coefficient of cos
            else
                KC = ones([N,1]) * [1,x(M+2:M+order)'] .* order_mat;        % coefficient of cos
                KS = ones([N,1]) * x(M+1+order:M+order*2)' .* order_mat;    % coefficient of sin
            end
            xcd = -2*pi * KC .* time_mat .* sin(theta_mat*2*pi) / N;    % cosine series partial freq
            xsd =  2*pi * KS .* time_mat .* cos(theta_mat*2*pi) / N;    % sine series partial freq


            A = [bits_patch(1:N,1:M),ones(N,1),xc(:,2:end),xs,sum(xcd+xsd,2)];
            b = -xc(:,1);
            x1 = linsolve(A,b);
            e1 = A*x1-b;

            A = [bits_patch(1:N,1:M),ones(N,1),xs(:,2:end),xc,sum(xcd+xsd,2)];
            b = -xs(:,1);
            x2 = linsolve(A,b);
            e2 = A*x2-b;
            
            if(rms(e1) < rms(e2))
                x = x1;
                sel = 0;
            else
                x = x2;
                sel = 1;
            end
            
            delta_f = x(end)*rate /N;
            relerr = rms(x(end)/N*A(:,end)) / sqrt(1+x(M+1+order)^2);

            fprintf('Freq fine iterating (%d): freq = %d, delta_f = %d, rel_err = %d\n',ii,freq,delta_f, relerr);

            if(relerr < reltol)
                break;
            end

        end

        warning on;
    end
    

    w0 = sqrt(1+x(M+1+order)^2);

    weight = x(1:M)'/w0.*(10.^-MAG);    % post-scaling
    weight = weight(max(L,1)).*K;       % calculate merged column weight
    
    offset = -x(M+1)/w0;
    
    postCal = weight*bits';
    
    if(sel)
        ideal = -(xs(:,1) + xs(:,2:end) * x(M+2:M+order) + xc * x(M+1+order:M+2*order))'/w0;
    else
        ideal = -(xc(:,1) + xc(:,2:end) * x(M+2:M+order) + xs * x(M+1+order:M+2*order))'/w0;
    end
    
    err = postCal-offset-ideal;

    if(sum(weight)<0)
        weight = -weight;
        offset = -offset;
        postCal = -postCal;
        ideal = -ideal;
        err = -err;
    end
    
    freqCal = freq;

end