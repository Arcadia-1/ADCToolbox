function [weight,offset,postCal,ideal,err,freqCal] = FGCalSine(bits,varargin)

    % FGCalSine — Foreground calibration using a sinewave input
    %
    % This function estimates per-bit weights and a DC offset for an ADC by
    % fitting the weighted sum of raw bit columns to a sine series at a given
    % (or estimated) normalized frequency Fin/Fs. It optionally performs a
    % coarse and fine frequency search to refine the input tone frequency.
    %
    % Notes:
    % - Data format: MSB-left to LSB-right in columns of `bits`.
    % - Each row of `bits` is one sample; each column is a bit/segment.
    % - If the file name does not match the function name in MATLAB, consider
    %   renaming the file to `FGCalSine.m` or the function to `FGCalSineMF`.
    
    % foreground calibration by inputting sinewave
    % data format: MSB left - LSB right
    % bits is the raw data, N row by M col, N is data points, M is bitwidth
    % freq is the relative test freq Fin/Fs
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
    % weight is normalized by MSB. i.e., weight[MSB] = 1. MSB defined as the first non-zero weight from left
    % offset is normalized by MSB too
    % postCal is the signal after calibration
    % ideal is the ideal sine signal 
    % err is the residue errors after calibration (excluding distortion)
    % freqCal is the fine tuned frequency from calibration
    
        [N,M] = size(bits);             % N: number of samples (rows), M: number of bit columns
        if(N < M)
            bits = bits';                % Ensure rows are samples, columns are bits
            [N,M] = size(bits);          % Update dimensions after transpose
        end
    
        % Parse optional inputs controlling frequency search and harmonic exclusion order
        p = inputParser;
        addOptional(p, 'freq', 0, @(x) isnumeric(x) && isscalar(x) && (x >= 0));             % normalized Fin/Fs (0 triggers frequency search)
        addOptional(p, 'rate', 0.5, @(x) isnumeric(x) && isscalar(x) && (x > 0) && (x < 1)); % adaptive rate for frequency updates
        addOptional(p, 'reltol', 1E-12, @(x) isnumeric(x) && isscalar(x) && (x > 0));        % stop criterion (relative error tolerance)
        addOptional(p, 'niter', 100, @(x) isnumeric(x) && isscalar(x) && (x > 0));           % max fine-search iterations
        addOptional(p, 'order', 1, @(x) isnumeric(x) && isscalar(x) && (x > 0));             % harmonics exclusion order (1 for no exclusion)
        addOptional(p, 'fsearch', 0, @(x) isnumeric(x) && isscalar(x));                      % force fine search (1) or not (0)
        addParameter(p, 'nomWeight', 2.^(M-1:-1:0));                                         % nominal bit weights (only effective when rank is deficient)
        parse(p, varargin{:});
        freq = p.Results.freq;            
        order = max(round(p.Results.order),1);  
        nomWeight = p.Results.nomWeight;  
        rate = p.Results.rate;            
        reltol = p.Results.reltol;        
        niter = p.Results.niter;          
        fsearch = p.Results.fsearch;      
        
        % Initialize link and scale tables used to map original columns to a potentially merged, rank-sufficient set of columns (bits_patch)
        L = [1:M];              % link from a column to its correlated column
        K = ones(1,M);          % weight ratio of a column to its correlated column
    
        % If columns (plus DC) are rank-deficient, try to patch by merging perfectly correlated columns and discarding constant ones.
        if(rank([bits,ones(N,1)]) < M+1)
            warning('Rank deficiency detected. Try patching...');
            bits_patch = [];      % deduplicated/merged bit columns used for fitting
            LR = [];              % reverse link from bits_patch to original column indices
            M2 = 0;               % number of columns in bits_patch
            for i1 = 1:M
                if(max(bits(:,i1))==min(bits(:,i1)))    % constant column -> no information and can be discarded
                    L(i1) = 0;
                    K(i1) = 0;
                elseif(rank([ones(N,1),bits_patch,bits(:,i1)]) > rank([ones(N,1),bits_patch]))  % column i1 adds rank -> keep it
                    bits_patch = [bits_patch,bits(:,i1)];
                    LR = [LR,i1];
                    [~,M2] = size(bits_patch);
                    L(i1) = M2;     % normal column: link to its own merged index
                else                % column i1 is correlated to the rest columns -> try to merge it into the existing column
                    flag = 0;       % flag to indicate if column i1 has been merged into the existing column
                    for i2 = 1:M2
                        r1 = bits(:,i1)-mean(bits(:,i1));         
                        r2 = bits_patch(:,i2) - mean(bits_patch(:,i2));
                        cor = mean(r1.*r2)/rms(r1)/rms(r2);       % correlation coefficient
                        if(abs(abs(cor)-1) < 1E-3)    % |cor|≈1 -> perfectly correlated (possibly inverted)
                            L(i1) = i2;                                % link column i1 to column i2
                            K(i1) = nomWeight(i1)/nomWeight(LR(i2));   % use nominal weight ratio
                            % Merge i1 into i2 to form a single effective column
                            bits_patch(:,i2) = bits_patch(:,i2) + bits(:,i1)*nomWeight(i1)/nomWeight(LR(i2));
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
            if(rank([ones(N,1),bits_patch]) < M+1)      % still deficient -> give up with guidance
                error('Patch failed: rank still deficient after patching. This may be fixed by changing nomWeight.')
            end
        else
            bits_patch = bits;                           % no patching needed
        end
    
        % Pre-scaling columns to avoid numerical conditioning problems in matrix solver
        MAG = floor(log10(max(abs([max(bits_patch);min(bits_patch)]))));  % column-wise base-10 magnitude
        MAG(isinf(MAG)) = 0;                                              % guard against inf (e.g., zeros)
        bits_patch = bits_patch.*(ones(N,1)*10.^(-MAG));                  % scale columns near unity
        
        % Coarse frequency search if freq not provided (freq==0)
        if(freq == 0)
            fsearch = 1;
            freq = [];
            for i1 = 1:min(M,5)     % first 5 columns are used to estimate the frequency - WARNING: this may not be a good practice for non-binary bit weighting
                fprintf('Freq coarse searching (%d/5):',i1);
                % Estimate Fin/Fs using a weighted sum of the top i1 columns
                freq = [freq, findFin(bits_patch(:,1:i1)*nomWeight(1:i1)')];
                fprintf(' freq = %d\n',freq(end));
            end
            freq = median(freq);    % use the median of the estimated frequencies
        end
        
        % Build harmonic basis and solve two sets of formulations to disambiguate
        theta_mat = (0:(N-1))'*freq*(1:order);   % phase matrix for harmonics 1..order
        xc = cos(theta_mat*2*pi);                % cosine basis (cols: harmonics)
        xs = sin(theta_mat*2*pi);                % sine basis (cols: harmonics)
        % Assumption 1: cosine is the unity basis
        A = [bits_patch(1:N,1:M),ones(N,1),xc(:,2:end),xs];
        b = -xc(:,1);
        x1 = linsolve(A,b);                      % solution vector for assumption 1
        % Assumption 2: sine is the unity basis
        A = [bits_patch(1:N,1:M),ones(N,1),xs(:,2:end),xc];
        b = -xs(:,1);
        x2 = linsolve(A,b);                      % solution vector for assumption 2
        if(rms(A*x1-b) < rms(A*x2-b))            % chooses the better solution based on the residual
            x = x1;                             
            sel = 0;                             % uses cosine-based solution   
        else
            x = x2;                             
            sel = 1;                             % uses sine-based solution
        end
        
        
        % Optional fine frequency search: augment matrix with d/d(freq) column and iterate
        if(fsearch)
    
            warning off;
    
            delta_f = 0;
            time_mat = (0:(N-1))'*ones([1,order]);   % time index matrix for derivative terms
            
            for ii = 1:niter
                freq = freq+delta_f;                % update frequency by prior correction
                theta_mat = (0:(N-1))'*freq*(1:order);          
                
                xc = cos(theta_mat*2*pi);  % cosine basis (cols: harmonics)
                xs = sin(theta_mat*2*pi);  % sine basis (cols: harmonics)
                
                order_mat = ones([N,1]) * (1:order);   % scale derivatives by harmonic index
                if(sel)
                    KS = ones([N,1]) * [1,x(M+2:M+order)'] .* order_mat;        % coefficients for sine terms
                    KC = ones([N,1]) * x(M+1+order:M+order*2)' .* order_mat;    % coefficients for cosine terms
                else
                    KC = ones([N,1]) * [1,x(M+2:M+order)'] .* order_mat;        % coefficients for cosine terms
                    KS = ones([N,1]) * x(M+1+order:M+order*2)' .* order_mat;    % coefficients for sine terms
                end
                % Partial derivatives w.r.t. frequency for each harmonic term
                xcd = -2*pi * KC .* time_mat .* sin(theta_mat*2*pi) / N;    % d/d(freq) of cosine series
                xsd =  2*pi * KS .* time_mat .* cos(theta_mat*2*pi) / N;    % d/d(freq) of sine series
    
    
                % Re-solve augmented systems with derivative column appended
                A = [bits_patch(1:N,1:M),ones(N,1),xc(:,2:end),xs,sum(xcd+xsd,2)];
                b = -xc(:,1);
                x1 = linsolve(A,b);
                e1 = A*x1-b;
    
                A = [bits_patch(1:N,1:M),ones(N,1),xs(:,2:end),xc,sum(xcd+xsd,2)];
                b = -xs(:,1);
                x2 = linsolve(A,b);
                e2 = A*x2-b;
                
                if(rms(e1) < rms(e2))
                    x = x1;       % prefer cosine-targeted update
                    sel = 0;
                else
                    x = x2;       % prefer sine-targeted update
                    sel = 1;
                end
                
                delta_f = x(end)*rate /N;                           % scaled frequency correction
                relerr = rms(x(end)/N*A(:,end)) / sqrt(1+x(M+1+order)^2);  % calculate the relative error
    
                fprintf('Freq fine iterating (%d): freq = %d, delta_f = %d, rel_err = %d\n',ii,freq,delta_f, relerr);
    
                if(relerr < reltol)
                    break;                   % stop if the relative error is less than the tolerance
                end
    
            end
    
            warning on;
        end
        
    
        w0 = sqrt(1+x(M+1+order)^2);        % magnitude of the fundamental
    
        weight = x(1:M)'/w0.*(10.^-MAG);    % undo column scaling and normalize by w0
        weight = weight(max(L,1)).*K;       % map merged weights back to original columns
        
        offset = -x(M+1)/w0;                % normalized DC offset 
        
        postCal = weight*bits';             % reconstructed calibrated signal
        
        if(sel)
            % If sine-based solution was chosen
            ideal = -(xs(:,1) + xs(:,2:end) * x(M+2:M+order) + xc * x(M+1+order:M+2*order))'/w0;
        else
            % If cosine-based solution was chosen
            ideal = -(xc(:,1) + xc(:,2:end) * x(M+2:M+order) + xs * x(M+1+order:M+2*order))'/w0;
        end
        
        err = postCal-offset-ideal;         % residual error after calibration
    
        % Enforce positive overall polarity: flip everything if total weight sum is negative
        if(sum(weight)<0)
            weight = -weight;
            offset = -offset;
            postCal = -postCal;
            ideal = -ideal;
            err = -err;
        end
        
        freqCal = freq;                     % return refined frequency estimate
    
    end
    