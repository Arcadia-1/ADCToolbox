function [weight,offset,postCal,ideal,err,freqCal] = FGCalSine(bits,varargin)

% FGCalSine — Foreground calibration using a sinewave input
%
% This function estimates per-bit weights and a DC offset for an ADC by
% fitting the weighted sum of raw bit columns to a sine series at a given
% (or estimated) normalized frequency Fin/Fs. It optionally performs a
% coarse and fine frequency search to refine the input tone frequency.
%
% Usage:
% [weight,offset,postCal,ideal,err,freqCal] = FGCalSine(bits[, Name, Value])
%
% Inputs:
% - bits: binary data as matrix (N row by M col, N is data points, M is bitwidth).
% - Each row of `bits` is one sample; each column is a bit/segment.
% - Name-Value arguments:
%   - freq: normalized frequency Fin/Fs (0 triggers frequency search), default is 0.
%   - rate: adaptive rate for frequency updates (0..1), default is 0.5.
%   - reltol: relative error tolerance, default is 1e-12.
%   - niter: max fine-search iterations, default is 100.
%   - order: harmonics exclusion order (1 for no exclusion), default is 1.
%   - fsearch: force fine search (1) or not (0), default is 0.
%   - verbose: print frequency search progress (1) or not (0), default is 0.
%   - nomWeight: nominal bit weights (only effective when rank is deficient), default is 2.^(M-1:-1:0).
%
% Multi-dataset extension:
%   If `bits` is a cell array {bits1,bits2,...} with per-dataset sine inputs
%   (possibly at different frequencies), this function will optionally search
%   the frequency of each dataset independently (when freq is unspecified/0),
%   then perform a single joint least-squares solve sharing one set of bit
%   weights and one DC offset across all datasets. Per-dataset harmonics are
%   modeled independently. Weight and offset are normalized by the magnitude of the first dataset's sinewave.
%  - usage: [weight,offset,postCal,ideal,err,freqCal] = FGCalSine({bits1, bits2, ...}, 'freq', [freq1, freq2, ...], ... );
%
% Outputs:
% - weight: the calibrated weights, normalized by the magnitude of sinewave. 
% - offset: the calibrated DC offset, normalized by the magnitude of sinewave.
% - postCal: the signal after calibration
% - ideal: the best fitted sinewave
% - err: the residue errors after calibration (excluding distortion)
% - freqCal: the fine tuned frequency from calibration

    % ==========================
    % Multi-dataset (cell) path
    % ==========================

    if iscell(bits)
        % Validate shapes and collect per-dataset sizes
        ND = numel(bits);   % Number of datasets
        if ND == 0
            error('FGCalSine:EmptyInput','Empty cell array for bits.');
        end
        bits_cell = cell(1,ND);
        Nk = zeros(1,ND);
        Mk = zeros(1,ND);
        for k = 1:ND
            Bk = bits{k};
            if isempty(Bk)
                error('FGCalSine:EmptyDataset','Dataset %d is empty.',k);
            end
            [nTmp,mTmp] = size(Bk);
            if nTmp < mTmp
                Bk = Bk';
                [nTmp,mTmp] = size(Bk);
            end
            bits_cell{k} = Bk;
            Nk(k) = nTmp;
            Mk(k) = mTmp;
        end
        if any(Mk ~= Mk(1))
            error('FGCalSine:InconsistentWidth','All datasets must have the same number of columns (bits).');
        end
        M_orig = Mk(1);

        % Parse options (allow vector freq); keep names consistent with single-dataset
        p = inputParser;
        addOptional(p, 'freq', 0, @(x) isnumeric(x) && isvector(x) && all(x>=0)); % scalar or vector (>=0)
        addOptional(p, 'rate', 0.5, @(x) isnumeric(x) && isscalar(x) && (x > 0) && (x < 1));
        addOptional(p, 'reltol', 1E-12, @(x) isnumeric(x) && isscalar(x) && (x > 0));
        addOptional(p, 'niter', 100, @(x) isnumeric(x) && isscalar(x) && (x > 0));
        addOptional(p, 'order', 1, @(x) isnumeric(x) && isscalar(x) && (x > 0));
        addOptional(p, 'fsearch', 0, @(x) isnumeric(x) && isscalar(x));
        addOptional(p, 'verbose', 0, @(x) isnumeric(x) && isscalar(x));
        addParameter(p, 'nomWeight', 2.^(M_orig-1:-1:0));
        parse(p, varargin{:});
        freq = p.Results.freq;
        order = max(round(p.Results.order),1);
        nomWeight = p.Results.nomWeight;
        rate = p.Results.rate;
        reltol = p.Results.reltol;
        niter = p.Results.niter;
        fsearch = p.Results.fsearch;
        verbose = p.Results.verbose;  

        % Normalize freq to vector length Kd
        if isscalar(freq)
            freq = ones(1,ND) * freq;
        end
        if numel(freq) ~= ND
            error('FGCalSine:FreqLength','Length of freq vector must match number of datasets.');
        end

        % Per-dataset frequency search (only for unknown entries)
        for k = 1:ND
            if freq(k) == 0 || fsearch == 1
                [~,~,~,~,~,fk] = FGCalSine(bits_cell{k}, 'freq', freq(k), 'fsearch', 1, ...
                    'order', order, 'rate', rate, 'reltol', reltol, 'niter', niter, 'nomWeight', nomWeight, 'verbose', verbose);
                freq(k) = fk;
            end
        end

        % Build a unified rank patch (L,K map) across all datasets by concatenation
        bits_all = vertcat(bits_cell{:});
        Ntot = size(bits_all,1);

        % Initialize link and scaling
        Lmap = 1:M_orig;
        Kmap = ones(1,M_orig);

        % Patching routine on concatenated data
        if(rank([bits_all,ones(Ntot,1)]) < M_orig+1)
            warning('Rank deficiency detected across datasets. Try patching...');
            bits_patch_all = [];
            LR = [];
            M2 = 0;
            for i1 = 1:M_orig
                if max(bits_all(:,i1))==min(bits_all(:,i1))
                    Lmap(i1) = 0;
                    Kmap(i1) = 0;
                elseif(rank([ones(Ntot,1),bits_patch_all,bits_all(:,i1)]) > rank([ones(Ntot,1),bits_patch_all]))
                    bits_patch_all = [bits_patch_all,bits_all(:,i1)];
                    LR = [LR,i1];
                    [~,M2] = size(bits_patch_all);
                    Lmap(i1) = M2;
                else
                    flag = 0;
                    for i2 = 1:M2
                        r1 = bits_all(:,i1)-mean(bits_all(:,i1));
                        r2 = bits_patch_all(:,i2)-mean(bits_patch_all(:,i2));
                        cor = mean(r1.*r2)/rms(r1)/rms(r2);
                        if(abs(abs(cor)-1) < 1E-3)
                            Lmap(i1) = i2;
                            Kmap(i1) = nomWeight(i1)/nomWeight(LR(i2));
                            bits_patch_all(:,i2) = bits_patch_all(:,i2) + bits_all(:,i1)*Kmap(i1);
                            flag = 1;
                            break;
                        end
                    end
                    if(flag == 0)
                        Lmap(i1) = 0;
                        Kmap(i1) = 0;
                        warning('Patch warning: cannot find correlated column for column %d. Resulting weight will be zero',i1);
                    end
                end
            end
            [~,M_patch] = size(bits_patch_all);
            if(rank([ones(Ntot,1),bits_patch_all]) < M_patch+1)
                error('Patch failed: rank still deficient after patching across datasets. Try adjusting nomWeight.');
            end
        else
            bits_patch_all = bits_all;
            M_patch = M_orig;
        end

        % Column magnitude scaling (from concatenated patched data)
        MAG = floor(log10(max(abs([max(bits_patch_all);min(bits_patch_all)]))));
        MAG(isinf(MAG)) = 0;
        bits_patch_all = bits_patch_all.*(ones(Ntot,1)*10.^(-MAG));

        numHCols = ND*order;
        xc = zeros(Ntot, numHCols);
        xs = zeros(Ntot, numHCols);

        rowStart = 1;
        for k = 1:ND
            Nk_k = Nk(k);
            rowEnd = rowStart + Nk_k - 1;

            theta_mat = (0:(Nk_k-1))' * freq(k) * (1:order);
            xc(rowStart:rowEnd, (k-1)*order+1 : k*order) = cos(theta_mat*2*pi);
            xs(rowStart:rowEnd, (k-1)*order+1 : k*order) = sin(theta_mat*2*pi);

            rowStart = rowEnd + 1;
        end

        % Assumption 1: cosine is the unity basis
        A = [bits_patch_all,ones(Ntot,1),xc(:,2:end),xs];
        b = -xc(:,1);
        x1 = linsolve(A,b);                      % solution vector for assumption 1
        % Assumption 2: sine is the unity basis
        A = [bits_patch_all,ones(Ntot,1),xs(:,2:end),xc];
        b = -xs(:,1);
        x2 = linsolve(A,b);                      % solution vector for assumption 2

        if(rms(A*x1-b) < rms(A*x2-b))            % chooses the better solution based on the residual
            x = x1;                             
            sel = 0;                             % uses cosine-based solution   
        else
            x = x2;                             
            sel = 1;                             % uses sine-based solution
        end

        % Normalization using dataset-1 quadrature fundamental
        w0 = sqrt(1 + x(M_patch + numHCols)^2);

        % Map weights back
        wpatch = (x(1:M_patch)'/w0) .* (10.^-MAG);
        weight = wpatch(max(Lmap,1)) .* Kmap;  % Lmap==0 -> Kmap==0 => weight=0
        offset = -x(M_patch+1)/w0;

        % Per-dataset signals
        postCal = cell(1,ND);
        ideal = cell(1,ND);
        err = cell(1,ND);

        if(sel)
            ideal{1} = -(xs(1:Nk(1),1) + xs(1:Nk(1),2:order) * x(M_patch + 1 + 1 : M_patch + 1 + order-1) + xc(1:Nk(1),1:order) * x(M_patch + 1 + numHCols : M_patch + 1 + numHCols + order-1))'/w0;
        else
            ideal{1} = -(xc(1:Nk(1),1) + xc(1:Nk(1),2:order) * x(M_patch + 1 + 1 : M_patch + 1 + order-1) + xs(1:Nk(1),1:order) * x(M_patch + 1 + numHCols : M_patch + 1 + numHCols + order-1))'/w0;
        end

        rowStart = Nk(1)+1;
        for k = 2:ND
            rowEnd = rowStart + Nk(k) - 1;
            if(sel)
                ideal{k} = -(xs(rowStart:rowEnd,(k-1)*order+1:k*order) * x(M_patch + 1 + (k-1)*order : M_patch + 1 + k*order - 1) +...
                             xc(rowStart:rowEnd,(k-1)*order+1:k*order) * x(M_patch + 1 + numHCols + (k-1)*order : M_patch + 1 + numHCols + k*order - 1) )'/w0;
            else
                ideal{k} = -(xc(rowStart:rowEnd,(k-1)*order+1:k*order) * x(M_patch + 1 + (k-1)*order : M_patch + 1 + k*order - 1) +...
                             xs(rowStart:rowEnd,(k-1)*order+1:k*order) * x(M_patch + 1 + numHCols + (k-1)*order : M_patch + 1 + numHCols + k*order - 1) )'/w0;
            end
            rowStart = rowEnd+1;
        end

        for k = 1:ND
            postCal{k} = weight * bits_cell{k}';
            err{k} = postCal{k} - offset - ideal{k};
        end

        % Enforce positive polarity
        if sum(weight) < 0
            weight = -weight;
            offset = -offset;
            for k = 1:ND
                postCal{k} = -postCal{k};
                ideal{k}   = -ideal{k};
                err{k}     = -err{k};
            end
        end

        freqCal = freq;
        return;
    end

    % ==================
    % Single-dataset path
    % ==================
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
    addOptional(p, 'verbose', 0, @(x) isnumeric(x) && isscalar(x));                      % print frequency search progress (1) or not (0)
    addParameter(p, 'nomWeight', 2.^(M-1:-1:0));                                         % nominal bit weights (only effective when rank is deficient)
    parse(p, varargin{:});
    freq = p.Results.freq;
    order = max(round(p.Results.order),1);
    nomWeight = p.Results.nomWeight;
    rate = p.Results.rate;
    reltol = p.Results.reltol;
    niter = p.Results.niter;
    fsearch = p.Results.fsearch;
    verbose = p.Results.verbose;      
    
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
            if(verbose)
                fprintf('Freq coarse searching (%d/5):',i1);
            end
            % Estimate Fin/Fs using a weighted sum of the top i1 columns
            freq = [freq, findFin(bits_patch(:,1:i1)*nomWeight(1:i1)')];
            if(verbose)
                fprintf(' freq = %d\n',freq(end));
            end
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

            if(verbose)
                fprintf('Freq fine iterating (%d): freq = %d, delta_f = %d, rel_err = %d\n',ii,freq,delta_f, relerr);
            end

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
