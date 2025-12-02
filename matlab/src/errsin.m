function [emean, erms, xx, anoi, pnoi, err, errxx] = errsin(sig, varargin)
%ERRSIN Analyze sinewave fit errors with histogram binning
%   This function fits a sinewave to the input signal and analyzes the
%   residual errors, binning them by either phase or value. It computes
%   mean and RMS errors per bin, and estimates amplitude and phase noise
%   components when using phase mode.
%
%   Syntax:
%     [emean, erms, xx, anoi, pnoi, err, errxx] = ERRSIN(sig)
%     [emean, erms, xx, anoi, pnoi, err, errxx] = ERRSIN(sig, Name, Value)
%
%   Inputs:
%     sig - Signal to be analyzed (typically ADC output or sinewave samples)
%       Vector of real numbers
%
%   Name-Value Arguments:
%     'bin' - Number of bins for histogram analysis
%       Positive integer (default: 100)
%     'fin' - Normalized input frequency (frequency/fs)
%       Positive scalar in range (0,1). If 0 or omitted, frequency is
%       automatically estimated (default: 0)
%     'disp' - Display plots of error analysis
%       Logical or numeric (0 or 1)
%       Default: nargout == 0 (auto-display when no outputs)
%     'xaxis' - X-axis mode for binning
%       'phase' = bin by phase angle (default)
%       'value' = bin by signal value
%     'erange' - Error range filter for output err and errxx
%       2-element vector [min, max] to filter x-axis values. If empty, no
%       filtering is applied (default: [])
%
%   Outputs:
%     emean - Mean error for each bin
%       Vector (1×bin)
%     erms - RMS error for each bin
%       Vector (1×bin)
%     xx - Bin centers, i.e., the x-axis for emean and erms
%       Vector (1×bin)
%       Phase mode: [0, 360), Value mode: centered bin values
%     anoi - Estimated amplitude noise RMS
%       Scalar. In phase mode, estimated from error pattern. In code mode,
%       returns NaN
%     pnoi - Estimated phase noise RMS (radians)
%       Scalar. In phase mode, normalized by signal magnitude. In code mode,
%       returns NaN
%     err - Raw errors for each sample point (sig_fit - sig)
%       Vector (N×1), filtered by erange if specified
%     errxx - X-axis values corresponding to err
%       Vector (N×1), same size as err
%       Phase mode: phase in degrees, Value mode: sig values
%
%   Examples:
%     % Basic error analysis with phase binning
%     sig = sin(2*pi*0.12345*(0:999)') + 0.01*randn(1000,1);
%     [emean, erms, xx, anoi, pnoi] = errsin(sig);
%
%     % Value mode analysis with 50 bins and turn on display
%     [emean, erms, xx] = errsin(sig, 'xaxis', 'value', 'bin', 50, 'disp', 1);
%
%     % Filter errors to specific phase range
%     [~, ~, ~, ~, ~, err, errxx] = errsin(sig, 'erange', [90, 180]);
%
%   Notes:
%     - Phase mode (xaxis = 'phase'):
%       * Bins errors by phase angle of the sinewave
%       * Estimates amplitude and phase noise using least squares fit
%     - Value mode (xaxis = 'value'):
%       * Bins errors by the signal value
%       * Useful for analyzing INL
%       * anoi and pnoi are set to NaN as they cannot be estimated
%     - The function automatically orients input signal to column vector
%     - If fin=0, sinfit automatically estimates the frequency
%
%   See also: sinfit, inlsin

    % Input validation
    if ~isreal(sig)
        error('errsin:invalidInput', 'Signal must be real.');
    end

    % Input parsing
    p = inputParser;
    addOptional(p, 'bin', 100, @(x) isnumeric(x) && isscalar(x) && (x > 0));
    addOptional(p, 'fin', 0, @(x) isnumeric(x) && isscalar(x) && (x > 0) && (x < 1));
    addOptional(p, 'disp', nargout == 0, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'xaxis', 'phase', @(x) ischar(x) && (strcmpi(x,'phase') || strcmpi(x,'value')));
    addParameter(p, 'erange', []);  % err filter, only the errors in erange are returned to err
    parse(p, varargin{:});
    bin = round(p.Results.bin);
    fin = p.Results.fin;
    disp = p.Results.disp;
    xaxis = lower(p.Results.xaxis);
    erange = p.Results.erange;

    % Ensure column vector orientation
    S = size(sig);
    if(S(1) < S(2))
        sig = sig';
    end

    % Fit sine wave to input signal
    if(fin == 0)
        [sig_fit,fin,mag,~,phi] = sinfit(sig);
    else
        [sig_fit,~,mag,~,phi] = sinfit(sig,fin);
    end

    % Calculate residual errors
    err = sig_fit-sig;

    % Value mode: bin by signal value
    if(strcmp(xaxis,'value'))
        errxx = sig;
        dat_min = min(sig);
        dat_max = max(sig);
        bin_wid = (dat_max-dat_min)/bin;
        xx = min(sig) + (1:bin)*bin_wid - bin_wid/2;

        enum = zeros(1,bin);
        esum = zeros(1,bin);
        erms = zeros(1,bin);

        % Accumulate errors for each bin
        for ii = 1:length(sig)
            b = min(floor((sig(ii)-dat_min)/bin_wid)+1,bin);
            esum(b) = esum(b) + err(ii);
            enum(b) = enum(b) + 1;
        end
        emean = esum./enum;

        % Calculate RMS error for each bin
        for ii = 1:length(sig)
            b = min(floor((sig(ii)-dat_min)/bin_wid)+1,bin);
            erms(b) = erms(b) + (err(ii) - emean(b))^2;
        end
        erms = sqrt(erms./enum);

        % No noise estimation in code mode
        anoi = nan;
        pnoi = nan;

        % Apply error range filter if specified
        if(~isempty(erange))
            eid = (errxx >= erange(1)) & (errxx <= erange(2));
            errxx = errxx(eid);
            err = err(eid);
        end

        % Display plots if requested
        if(disp)
            subplot(2,1,1);

            plot(sig,err,'r.');
            hold on;
            plot(xx,emean,'b-');
            axis([dat_min,dat_max,min(err),max(err)]);
            ylabel('error');
            xlabel('value');

            if(~isempty(erange))
                plot(errxx,err,'m.');
            end

            legend('error','emean');
            
            subplot(2,1,2);
            bar(xx,erms);
            axis([dat_min,dat_max,0,max(erms)*1.1]);
            xlabel('value');
            ylabel('RMS error');


        end

    % Phase mode: bin by phase angle
    else
        errxx = mod(phi/pi*180 + (0:length(sig)-1)*fin*360,360);
        xx = (0:bin-1)/bin*360;

        enum = zeros(1,bin);
        esum = zeros(1,bin);
        erms = zeros(1,bin);

        % Accumulate errors for each phase bin
        for ii = 1:length(sig)
            b = mod(round(errxx(ii)/360*bin),bin)+1;
            esum(b) = esum(b) + err(ii);
            enum(b) = enum(b) + 1;
        end
        emean = esum./enum;

        % Calculate RMS error for each phase bin
        for ii = 1:length(sig)
            b = mod(round(errxx(ii)/360*bin),bin)+1;
            erms(b) = erms(b) + (err(ii) - emean(b))^2;
        end
        erms = sqrt(erms./enum);

        % Estimate amplitude and phase noise components
        % Amplitude noise affects all phases equally (cos^2 pattern)
        % Phase noise creates errors proportional to slope (sin^2 pattern)
        asen = abs(cos(xx/360*2*pi)).^2;    % amplitude noise sensitivity
        psen = abs(sin(xx/360*2*pi)).^2;    % phase noise sensitivity

        % Least squares fit: erms^2 = anoi^2*asen + (pnoi*mag)^2*psen + baseline
        tmp = linsolve([asen',psen',ones(bin,1)], erms'.^2);

        anoi = sqrt(tmp(1));
        pnoi = sqrt(tmp(2))/mag;
        ermsbl = tmp(3);    % erms baseline

        % Handle negative or complex results (physical constraint violation)
        if(anoi < 0 || imag(anoi) ~= 0)     % Try phase noise only
            tmp = linsolve([psen',ones(bin,1)], erms'.^2);
            anoi = 0;
            pnoi = sqrt(tmp(1))/mag;
            ermsbl = tmp(2);

            if(pnoi < 0 || imag(pnoi) ~= 0)  % Neither fit works
                anoi = 0;
                pnoi = 0;
                ermsbl = mean(erms.^2);
            end
        end

        if(pnoi < 0 || imag(pnoi) ~= 0)     % Try amplitude noise only
            tmp = linsolve([asen',ones(bin,1)], erms'.^2);
            pnoi = 0;
            anoi = sqrt(tmp(1));
            ermsbl = tmp(2);

            if(anoi < 0 || imag(anoi) ~= 0)  % Neither fit works
                anoi = 0;
                pnoi = 0;
                ermsbl = mean(erms.^2);
            end
        end

        % Apply error range filter if specified
        if(~isempty(erange))
            eid = (errxx >= erange(1)) & (errxx <= erange(2));
            errxx = errxx(eid);
            err = err(eid);
        end

        % Display plots if requested
        if(disp)
            subplot(2,1,1);

            yyaxis left;
            plot(errxx,sig,'k.');
            axis([0,360,min(sig),max(sig)]);
            ylabel('data');

            yyaxis right;
            plot(errxx,err,'r.');
            hold on;
            plot(xx,emean,'b-');
            axis([0,360,min(err),max(err)]);
            ylabel('error');

            legend('data','error');
            xlabel('phase(deg)');

            if(~isempty(erange))
                plot(errxx,err,'m.');
            end

            subplot(2,1,2);
            bar(xx,erms);
            hold on;
            plot(xx, sqrt((anoi.^2)*asen + ermsbl), 'b-', 'LineWidth',2);
            plot(xx, sqrt((pnoi.^2)*psen*(mag^2) + ermsbl), 'r-', 'LineWidth',2);
            axis([0,360,0,max(erms)*1.2]);
            text(10, max(erms)*1.15, sprintf('Normalized Amplitude Noise RMS = %.2d',anoi/mag), 'color', [0,0,1]);
            text(10, max(erms)*1.05, sprintf('Phase Noise RMS = %.2d rad',pnoi), 'color', [1,0,0]);
            xlabel('phase(deg)');
            ylabel('RMS error');
        end
    end

end
