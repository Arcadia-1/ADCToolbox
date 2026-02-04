function [emean, erms, xx, anoi, pnoi, err, errxx, R2] = errsin_hp(sig, varargin)
%ERRSIN_HP High-precision sinewave error analysis (Raw Regression + Binned R2)
%   Performs decomposition of residual errors into Amplitude Noise (AM) and 
%   Phase Noise (PM) using high-precision Least Squares regression on raw data.
%
%   [ALGORITHM STRATEGY]
%   1. ESTIMATION: Uses RAW data (err^2) for maximum statistical precision.
%   2. EVALUATION: Uses BINNED data (rms^2) for calculating R2 goodness-of-fit.
%      This avoids low R2 caused by raw noise variance, providing a metric
%      that reflects how well the model captures the noise trend.
%
%   [PHYSICS DEFINITION - ADAPTED FOR COSINE FIT]
%   Since sinfit returns A*cos(wt+phi):
%   - Amplitude Noise (AM): Max at Peaks (0, 180, 360). Correlates with cos^2.
%   - Phase Noise (PM): Max at Zero-Cross (90, 270). Correlates with sin^2.
%
%   Syntax:
%     [emean, erms, xx, anoi, pnoi, err, errxx, R2] = ERRSIN_HP(sig, ...)
%
%   Inputs:
%     sig - Real input signal vector
%     Name-Value Pairs: 'bin', 'fin', 'disp', 'xaxis', 'erange'
%
%   Outputs:
%     anoi - Amplitude Noise RMS (Volts)
%     pnoi - Phase Noise RMS (Radians)
%     R2   - Goodness of Fit based on Binned RMS Trend (0.0 to 1.0).
%
%   See also: SINFIT

    % --- 1. Input Validation & Parsing ---
    if ~isreal(sig), error('errsin_hp:invalidInput', 'Signal must be real.'); end
    
    p = inputParser;
    addOptional(p, 'bin', 100, @(x) isnumeric(x) && isscalar(x) && (x > 0));
    addOptional(p, 'fin', 0, @(x) isnumeric(x) && isscalar(x) && (x > 0) && (x < 1));
    addOptional(p, 'disp', nargout == 0, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'xaxis', 'phase', @(x) ischar(x) && (strcmpi(x,'phase') || strcmpi(x,'value')));
    addParameter(p, 'erange', []);
    parse(p, varargin{:});
    
    bin = round(p.Results.bin);
    fin = p.Results.fin;
    is_disp = p.Results.disp;
    xaxis = lower(p.Results.xaxis);
    erange = p.Results.erange;
    if size(sig,1) < size(sig,2), sig = sig'; end

    % --- 2. Sine Fit (Returns Cosine Phase) ---
    if(fin == 0)
        [sig_fit,fin,mag,~,phi] = sinfit(sig);
    else
        [sig_fit,~,mag,~,phi] = sinfit(sig,fin);
    end
    err = sig_fit - sig;

    % --- 3. Mode Handling ---
    if(strcmp(xaxis,'value'))
        % [Value Mode - Legacy Code]
        errxx = sig;
        dat_min = min(sig); dat_max = max(sig);
        bin_wid = (dat_max-dat_min)/bin;
        xx = dat_min + (1:bin)*bin_wid - bin_wid/2;
        bin_idx = floor((sig - dat_min)/bin_wid) + 1;
        bin_idx(bin_idx < 1) = 1; bin_idx(bin_idx > bin) = bin;
        esum = accumarray(bin_idx, err, [bin 1]);
        enum = accumarray(bin_idx, 1, [bin 1]);
        emean = (esum ./ max(1, enum))';
        err_dev = (err - emean(bin_idx)').^2;
        erms_sq = accumarray(bin_idx, err_dev, [bin 1]) ./ max(1, enum);
        erms = sqrt(erms_sq)';
        anoi = nan; pnoi = nan; R2 = nan;
        if is_disp, plot_value_mode(sig, err, xx, emean, erms, dat_min, dat_max); end
        
    else
        % --- 4. High Precision Phase Mode ---
        N = length(sig);
        phase_raw_rad = 2*pi*fin*(0:N-1)' + phi;
        phase_deg = mod(phase_raw_rad * 180/pi, 360);
        errxx = phase_deg;
        xx = (0:bin-1)/bin*360; 
        
        % --- A. CORE ALGORITHM: Raw Regression (For Estimation Accuracy) ---
        phase_rad = mod(phase_raw_rad, 2*pi);
        
        % PHYSICS DEFINITION:
        % AM -> cos^2 (Peaks)
        % PM -> sin^2 (Zero Cross)
        asen_raw = cos(phase_rad).^2; 
        psen_raw = sin(phase_rad).^2;
        
        % Solve: err^2 = c_am * cos^2 + c_pm * sin^2
        A_matrix = [asen_raw, psen_raw];
        b_vec = err.^2;
        coeffs = A_matrix \ b_vec;
        
        % Extract Raw Results
        var_am = max(0, coeffs(1));
        var_pm = max(0, coeffs(2));
        anoi = sqrt(var_am);
        pnoi = sqrt(var_pm) / mag;
        
        % --- B. Visualization Binning & R2 Calculation (For Trend Evaluation) ---
        bin_idx = floor(errxx / 360 * bin) + 1;
        bin_idx(bin_idx > bin) = bin;
        
        % Calculate Binned Statistics (RMS^2) using accumarray for speed
        esum = accumarray(bin_idx, err, [bin 1]);
        enum = accumarray(bin_idx, 1, [bin 1]);
        emean = (esum ./ max(1, enum))';
        
        err_dev = (err - emean(bin_idx)').^2;
        erms_sq_binned = accumarray(bin_idx, err_dev, [bin 1]) ./ max(1, enum);
        erms = sqrt(erms_sq_binned)';
        
        % Calculate R2 based on Binned Data (Trend Matching)
        % Generate model prediction at bin centers
        rad_bin = xx / 180 * pi;
        asen_bin = cos(rad_bin).^2;
        psen_bin = sin(rad_bin).^2;
        
        % The model curve derived from raw fit
        model_power = coeffs(1)*asen_bin + coeffs(2)*psen_bin;
        
        % Compare Binned Data vs Model
        % Only consider bins that actually have data
        valid_mask = (enum > 0)';
        y_data = erms_sq_binned(valid_mask)';
        y_fit  = model_power(valid_mask);
        
        if isempty(y_data)
            R2 = 0;
        else
            SS_res = sum((y_data - y_fit).^2);
            SS_tot = sum((y_data - mean(y_data)).^2);
            if SS_tot < 1e-20
                R2 = 1.0; % Perfect flat line match (e.g., pure thermal noise match)
            else
                R2 = 1 - (SS_res / SS_tot);
            end
        end
        
        % Filter output if needed
        if ~isempty(erange)
            mask = (errxx >= erange(1)) & (errxx <= erange(2));
            errxx = errxx(mask); err = err(mask);
        end
        
        % Plotting
        if is_disp, plot_phase_mode(sig, err, errxx, xx, emean, erms, anoi, pnoi, mag, R2); end
    end
end

% --- Plotting Functions ---
function plot_value_mode(sig, err, xx, emean, erms, dmin, dmax)
    subplot(2,1,1); plot(sig, err, 'r.', 'MarkerSize', 1); hold on;
    plot(xx, emean, 'b-', 'LineWidth', 1.5);
    axis([dmin, dmax, min(err), max(err)]); ylabel('Error (V)'); xlabel('Value (V)');
    subplot(2,1,2); bar(xx, erms, 'FaceColor', [0.7 0.7 0.8], 'EdgeColor', 'none');
    axis([dmin, dmax, 0, max(erms)*1.1]); xlabel('Value (V)'); ylabel('RMS Error (V)');
end

function plot_phase_mode(sig, err, errxx, xx, emean, erms, anoi, pnoi, mag, R2)
    % Formatting helper
    scale_u = 1e6; unit = 'u';
    if max(erms) > 1e-3, scale_u = 1e3; unit = 'm'; end
    
    subplot(2,1,1); yyaxis left; plot(errxx, sig, 'k.', 'MarkerSize', 1, 'Color', [0.8 0.8 0.8]);
    ylabel('Signal'); axis tight; yyaxis right; plot(errxx, err*scale_u, 'r.', 'MarkerSize', 1); hold on;
    plot(xx, emean*scale_u, 'b-', 'LineWidth', 1.5); ylabel(['Error (' unit 'V)']); 
    xlim([0 360]); title('Raw Residual Errors');
    
    subplot(2,1,2); bar(xx, erms*scale_u, 'FaceColor', [0.85 0.85 0.95], 'EdgeColor', 'none'); hold on;
    rad_plot = xx/360*2*pi;
    
    % Plot curves matching the Cosine basis physics
    asen_plot = cos(rad_plot).^2;
    psen_plot = sin(rad_plot).^2;
    am_curve = sqrt(anoi^2 * asen_plot);
    pm_curve = sqrt((pnoi*mag)^2 * psen_plot);
    total_curve = sqrt(am_curve.^2 + pm_curve.^2);
    
    plot(xx, total_curve*scale_u, 'k--', 'LineWidth', 1.5);
    plot(xx, am_curve*scale_u, 'b-', 'LineWidth', 2);
    plot(xx, pm_curve*scale_u, 'r-', 'LineWidth', 2);
    
    xlim([0 360]); ylim([0, max(erms*scale_u)*1.3]); 
    xlabel('Phase (deg)'); ylabel(['RMS Error (' unit 'V)']);
    
    txt = { sprintf('AM: %.2f %sV', anoi*scale_u, unit), ...
            sprintf('PM: %.2f %srad', pnoi*scale_u, unit), ...
            sprintf('R^2: %.3f (Trend)', R2) };
            
    text(10, max(erms*scale_u)*1.1, txt, 'BackgroundColor','w', 'EdgeColor', 'k');
    legend('Meas RMS','Total Fit','AM (cos^2)','PM (sin^2)','Location','northeast');
    grid on;
end