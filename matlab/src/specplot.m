function [enob,sndr,sfdr,snr,thd,sigpwr,noi,nsd,h] = specplot(sig,varargin)
%SPECPLOT Plot power spectrum and calculate ADC performance metrics
%   This function performs spectral analysis on ADC data and calculates key
%   performance metrics including ENOB, SNDR, SFDR, SNR, and THD. It supports
%   various windowing functions, oversampling ratio (OSR), coherent averaging,
%   and customizable plotting options.
%
%   Syntax:
%     [enob,sndr,sfdr,snr,thd,sigpwr,noi,nsd,h] = SPECPLOT(sig)
%     [enob,sndr,sfdr,snr,thd,sigpwr,noi,nsd,h] = SPECPLOT(sig, Fs)
%     [enob,sndr,sfdr,snr,thd,sigpwr,noi,nsd,h] = SPECPLOT(sig, Fs, maxCode)
%     [enob,sndr,sfdr,snr,thd,sigpwr,noi,nsd,h] = SPECPLOT(sig, Fs, maxCode, harmonic)
%     [enob,sndr,sfdr,snr,thd,sigpwr,noi,nsd,h] = SPECPLOT(sig, 'Name', Value)
%
%   Inputs:
%     sig - Signal to be plot, typically the ADC's output data
%       Vector or Matrix (N_run x N_fft)
%       Each row represents a separate measurement run for averaging
%
%   Optional Positional Inputs:
%     Fs - Sampling frequency in Hz. Default: 1
%       Scalar, positive real number
%     maxCode - Full scale range (max-min). Default: max(sig)-min(sig)
%       Scalar, positive real number
%     harmonic - Number of harmonics to analyze. Default: 5
%       Scalar, positive integer
%       Set negative to exclude harmonics from noise calculation
%
%   Name-Value Parameters:
%     'OSR' - Oversampling ratio. Default: 1
%       Scalar, positive real number
%       Defines signal bandwidth as Fs/(2*OSR)
%     'window' - Window function. Default: 'hann'
%       String: 'hann' (Hanning window) or 'rect' (Rectangle window)
%       Function handle: e.g., @hann, @blackman, @rectwin (requires Signal Processing Toolbox)
%       Alias: 'winType' (deprecated, use 'window')
%     'maxSignal' - Full scale range (max-min). Default: max(sig)-min(sig)
%       Scalar, positive real number
%       Alias: 'maxCode' (deprecated, use 'maxSignal')
%     'sideBin' - Number of bins around signal peak to include. Default: 1
%       Scalar, non-negative integer
%     'label' - Enable plot annotations. Default: 1
%       0 or 1
%     'assumedSignal' - Override signal power in dB. Default: NaN
%       Scalar, real number or NaN
%     'disp' - Enable plotting. Default: 1
%       0 or 1
%       Alias: 'isPlot' (deprecated, use 'disp')
%     'cutoff' - Cutoff frequency for low-frequency noise removal in Hz. Default: 0
%       Scalar, non-negative real number
%       Alias: 'noFlicker' (deprecated, use 'cutoff')
%     'nTHD' - Number of harmonics for THD calculation. Default: 5
%       Scalar, positive integer
%     'coAvg' - Enable coherent averaging. Default: 0
%       0 or 1
%       Uses phase alignment before averaging
%     'NFMethod' - Noise floor estimation method. Default: 'median'
%       String: 'median' (median-based), 'mean' (trimmed mean), 'exclude' (exclude harmonics)
%       Number: 0 (median-based), 1 (trimmed mean), 2 (exclude harmonics)
%
%   Outputs:
%     enob - Effective Number of Bits
%       Scalar, real number
%       Calculated as (sndr-1.76)/6.02
%     sndr - Signal-to-Noise and Distortion Ratio in dB
%       Scalar, real number
%     sfdr - Spurious-Free Dynamic Range in dB
%       Scalar, real number
%     snr - Signal-to-Noise Ratio in dB
%       Scalar, real number
%     thd - Total Harmonic Distortion in dB
%       Scalar, real number
%     sigpwr - Signal power in dBFS
%       Scalar, real number
%     noi - Noise Floor in dB
%       Scalar, real number
%     nsd - Noise Spectral Density in dBFS/Hz
%       Scalar, real number
%     h - Plot handle or empty array if isPlot=0
%       Graphics handle or []
%
%   Examples:
%     % Basic usage with default parameters (uses built-in Hanning window)
%     sig = sin(2*pi*0.1*(0:1023)) + randn(1,1024)*0.01;
%     [enob,sndr,sfdr] = specplot(sig);
%
%     % Specify sampling frequency and full scale
%     [enob,sndr,sfdr] = specplot(sig, 100e6, 2^16);
%
%     % Use oversampling with rectangle window (no toolbox required)
%     [enob,sndr,sfdr] = specplot(sig, 'OSR', 32, 'window', 'rect');
%
%     % Use other window functions (requires Signal Processing Toolbox)
%     [enob,sndr,sfdr] = specplot(sig, 'OSR', 32, 'window', @blackman);
%
%     % Use trimmed mean for noise floor estimation
%     [enob,sndr,sfdr] = specplot(sig, 'NFMethod', 'mean');
%
%     % Multiple runs with coherent averaging
%     sig = randn(10, 1024);  % 10 runs of 1024 samples
%     [enob,sndr,sfdr] = specplot(sig, 'coAvg', 1);
%
%     % Disable plotting and use assumed signal power
%     [enob,sndr,sfdr] = specplot(sig, 'disp', 0, 'assumedSignal', -3);
%
%   Notes:
%     - Signal can be provided as a row vector, column vector, or matrix
%     - For matrix input, each row is treated as a separate measurement
%     - The FFT length is determined by the number of columns (or rows if column vector)
%     - Coherent averaging ('coAvg') aligns phase before averaging to lower noise floor
%     - The noise floor is calculated within the signal band [0, Fs/(2*OSR)]
%     - Setting harmonic < 0 removes harmonics from both the analysis and display
%     - dBFS = 0 referred to a full-scale sinewave signal
%     - Built-in windows ('hann', 'rect') do not require Signal Processing Toolbox
%     - Custom window function handles (e.g., @blackman) require Signal Processing Toolbox
%     - NFMethod 'median' validates noise distribution and warns if significantly non-exponential
%
%   See also: alias, sinfit, fft, window

% Input parsing and validation
p = inputParser;
validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
validWindow = @(x) (ischar(x) && ismember(x, {'hann', 'rect'})) || isa(x, 'function_handle');
validNFMethod = @(x) (isnumeric(x) && ismember(x, [0, 1, 2])) || (ischar(x) && ismember(x, {'median', 'mean', 'exclude'}));
addOptional(p, 'Fs', 1, validScalarPosNum);
addOptional(p, 'maxCode', max(max(sig))-min(min(sig)), validScalarPosNum);
addOptional(p, 'harmonic', 5);
addParameter(p, 'OSR', 1, validScalarPosNum);
% Old parameter names (for backward compatibility)
addParameter(p, 'winType', 'hann', validWindow);
addParameter(p, 'isPlot', 1, @(x)ismember(x, [0, 1]));
addParameter(p, 'noFlicker', 0, validScalarPosNum);
% New parameter names (aliases with higher priority)
addParameter(p, 'window', 'hann', validWindow);
addParameter(p, 'maxSignal', NaN, validScalarPosNum);
addParameter(p, 'disp', NaN, @(x)ismember(x, [0, 1]));
addParameter(p, 'cutoff', 0, validScalarPosNum);
% Other parameters
addParameter(p, 'sideBin', 1, @(x) isnumeric(x) && isscalar(x) && (x >= 0));
addParameter(p, 'label', 1, @(x)ismember(x, [0, 1]));
addParameter(p, 'assumedSignal', NaN);
addParameter(p, 'nTHD', 5, validScalarPosNum);
addParameter(p, 'coAvg', 0, @(x)ismember(x, [0, 1]));
addParameter(p, 'NFMethod', 'median', validNFMethod);
parse(p, varargin{:});

% Extract parsed parameters
Fs = p.Results.Fs;
harmonic = p.Results.harmonic;
OSR = p.Results.OSR;
sideBin = p.Results.sideBin;
label = p.Results.label;
assumedSignal = p.Results.assumedSignal;
nTHD = p.Results.nTHD;
coAvg = p.Results.coAvg;

% Convert NFMethod from string to numeric if needed
if ischar(p.Results.NFMethod)
    switch p.Results.NFMethod
        case 'median'
            nfmethod = 0;
        case 'mean'
            nfmethod = 1;
        case 'exclude'
            nfmethod = 2;
    end
else
    nfmethod = p.Results.NFMethod;
end

% Handle aliased parameters (new names override old names)
% maxSignal/maxCode
if ~isnan(p.Results.maxSignal)
    maxSignal = p.Results.maxSignal;  % New name has priority
else
    maxSignal = p.Results.maxCode;  % Fall back to old name
end

% window/winType - use windowFunc as internal variable to avoid shadowing built-in
if ~isequal(p.Results.window, 'hann')
    windowFunc = p.Results.window;  % New name has priority if not default
else
    windowFunc = p.Results.winType;  % Fall back to old name
end

% disp/isPlot
if ~isnan(p.Results.disp)
    dispPlot = p.Results.disp;  % New name has priority (use dispPlot to avoid conflict)
else
    dispPlot = p.Results.isPlot;  % Fall back to old name
end

% cutoff/noFlicker
if p.Results.cutoff > 0
    cutoffFreq = p.Results.cutoff;  % New name has priority
else
    cutoffFreq = p.Results.noFlicker;  % Fall back to old name
end

% Determine data dimensions and FFT length
% Convert column vector to row vector if needed
[N,M] = size(sig);
N_fft = M;
if(M==1 && N > 1)
    sig = sig';
    N_fft = N;
end

[N_run,~] = size(sig);

% Calculate number of positive frequency bins
Nd2 = floor(N_fft/2)+1;

% Generate frequency axis
freq = (0:(Nd2-1))/N_fft*Fs;

% Generate window function
if ischar(windowFunc)
    % Use embedded window functions (no toolbox required)
    if strcmp(windowFunc, 'hann')
        win = hannwin(N_fft);
    elseif strcmp(windowFunc, 'rect')
        win = rectwin_emb(N_fft);
    else
        win = rectwin_emb(N_fft);
        warning("Unsupported window type '%s', using rectangle window", windowFunc);
    end
else
    % Use function handle (requires Signal Processing Toolbox)
    try
        win = window(windowFunc,N_fft,'periodic')';
    catch
        try
            win = window(windowFunc,N_fft)';
        catch
            win = rectwin_emb(N_fft);
            warning("Unsupported window function, using rectangle window");
        end
    end
end

% Initialize spectrum accumulator and measurement counter
spec = zeros([1,N_fft]);
ME = 0;
for iter = 1:N_run
    tdata = sig(iter,:);
    % Skip empty data
    if(rms(tdata)==0)
        continue;
    end
    % Normalize to full scale, remove DC, and apply window
    tdata = tdata./maxSignal;
    tdata = tdata-mean(tdata);
    tdata = tdata.*win/sqrt(mean(win.^2));

    if(coAvg)
        % Coherent averaging: align phase before averaging
        tspec = fft(tdata);
        tspec(1) = 0;  % Remove DC component
        % Find fundamental signal bin in signal band
        [~, bin] = max(abs(tspec(1:floor(N_fft/2/OSR))));
        phi = tspec(bin)/abs(tspec(bin));  % Extract phase of fundamental

        % Phase alignment: rotate spectrum to align fundamental phase
        phasor = conj(phi);
        marker = zeros(1,N_fft);
        % Apply phase shift to harmonics (accounting for aliasing)
        for iter2 = 1:N_fft
            J = (bin-1)*iter2;
            % Determine if harmonic is in even or odd Nyquist zone
            if(mod(floor(J/N_fft*2),2) == 0)
                % Even zone: normal aliasing
                b = J-floor(J/N_fft)*N_fft+1;
                if(marker(b) == 0)
                    tspec(b) = tspec(b).*phasor;
                    marker(b) = 1;
                end
            else
                % Odd zone: mirrored aliasing
                b = N_fft-J+floor(J/N_fft)*N_fft+1;
                if(marker(b) == 0)
                    tspec(b) = tspec(b).*conj(phasor);
                    marker(b) = 1;
                end
            end
            phasor = phasor * conj(phi);
        end

        % Apply phase shift to non-harmonic components
        for iter2 = 1:N_fft
            if(marker(iter2) == 0)
                tspec(iter2) = tspec(iter2).*(conj(phi).^((iter2-1)/(bin-1)));
            end
        end

        spec = spec + tspec;  % Coherent sum

    else
        % Power averaging: accumulate power spectrum
        spec = spec+abs(fft(tdata)).^2;
    end

    ME = ME+1;
end

% Normalize spectrum based on averaging method
if(coAvg)
    % Coherent averaging: take magnitude squared after sum, scale by number of runs
    spec = abs(spec).^2/(N_fft^2)*16/ME^2;
else
    % Power averaging: scale by number of runs
    spec(1) = 0;  % Remove DC
    spec = spec/(N_fft^2)*16/ME;
end
spec = spec(1:Nd2);  % Keep only positive frequencies
spec_inband = spec(1:floor(N_fft/2/OSR));  % Extract signal band


% Remove flicker noise (1/f noise) if requested
if cutoffFreq > 0
    spec(1:ceil(cutoffFreq/Fs*N_fft)) = 0;
end

% Find signal bin and refine using parabolic interpolation
[~, bin] = max(spec_inband);
sig_e = log10(spec(bin));
sig_l = log10(spec(min(max(bin-1,1),Nd2)));
sig_r = log10(spec(min(max(bin+1,1),Nd2)));
% Parabolic interpolation for sub-bin frequency accuracy
bin_r = bin + (sig_r-sig_l)/(2*sig_e-sig_l-sig_r)/2;
if(isnan(bin_r))
    bin_r = bin;
end

% Calculate signal power including side bins
sig = sum(spec(max(bin-sideBin,1):min(bin+sideBin,floor(N_fft/2/OSR))));
pwr = 10*log10(sig);
% Override with assumed signal power if provided
if(~isnan(assumedSignal))
    sig = 10.^(assumedSignal/10);
    pwr = assumedSignal;
end

% Remove harmonics from spectrum for display if harmonic < 0
if(harmonic < 0)
    for i = 2:-harmonic
        b = alias(round((bin_r-1)*i),N_fft);
        spec(max(b+1-sideBin,1):min(b+1+sideBin,Nd2)) = 0;
    end
end

% Plot spectrum if requested
if(dispPlot)
    % Use linear or log scale depending on OSR
    if (OSR == 1)
        h = plot(freq,10*log10(spec+10^(-20)));
    else
        h = semilogx(freq,10*log10(spec+10^(-20)));
    end

    grid on;
    hold on;
    % Mark signal bins if label enabled
    if(label)
        if (OSR == 1)
            plot(freq(max(bin-sideBin,1):min(bin+sideBin,Nd2)),10*log10(spec(max(bin-sideBin,1):min(bin+sideBin,Nd2))),'r-','linewidth',0.5);
            plot(freq(bin),10*log10(spec(bin)),'ro','linewidth',0.5);
        else
            semilogx(freq(max(bin-sideBin,1):min(bin+sideBin,Nd2)),10*log10(spec(max(bin-sideBin,1):min(bin+sideBin,Nd2))),'r-','linewidth',0.5);
        end
    end
    % Mark harmonics if requested
    if(harmonic > 0)
        for i = 2:harmonic
            b = alias(round((bin_r-1)*i),N_fft);
            plot(b/N_fft*Fs,10*log10(spec(b+1)+10^(-20)),'rs');
            text(b/N_fft*Fs,10*log10(spec(b+1)+10^(-20))+5,num2str(i),'fontname','Arial','fontsize',12,'horizontalalignment','center');
        end
    end
end

% Calculate SNDR and SFDR
% Save signal bin value for SFDR calculation
sigs = spec(bin);
if(~isnan(assumedSignal))
    sigs = 10.^(assumedSignal/10);
end
% Remove signal and DC from spectrum for noise/distortion calculation
spec(max(bin-sideBin,1):min(bin+sideBin,Nd2)) = 0;
spec(1:sideBin) = 0;
spec_inband = spec(1:floor(N_fft/2/OSR));
noi = sum(spec_inband);  % Total noise + distortion power

% Find largest spur for SFDR
[spur, sbin] = max(spec_inband);
SNDR = 10*log10(sig/noi);
SFDR = 10*log10(sigs/spur);
ENoB = (SNDR-1.76)/6.02;

% Mark maximum spur on plot
if(dispPlot && label)
    plot((sbin-1)/N_fft*Fs,10*log10(spur+10^(-20)),'rd');
    text((sbin-1)/N_fft*Fs,10*log10(spur+10^(-20))+5,'MaxSpur','fontname','Arial','fontsize',10,'horizontalalignment','center');
end

% Calculate noise floor using selected method
if(nfmethod == 0)
    % Method 0: Median-based estimation (robust to spurs)
    df = 2*N_run;
    noi = median(spec(1:floor(N_fft/2/OSR)))/sqrt((1-2/(9*df))^3) *floor(N_fft/2/OSR);
elseif(nfmethod == 1)
    % Method 1: Trimmed mean (removes top/bottom 5%)
    spec_sort = sort(spec(1:floor(N_fft/2/OSR)));
    noi = mean(spec_sort(floor(N_fft/2/OSR*0.05):floor(N_fft/2/OSR*0.95)))*floor(N_fft/2/OSR);
else
    % Method 2: Exclude harmonics from noise calculation
    spec_noise = spec;
    for i = 2:nTHD
        b = alias(round((bin_r-1)*i),N_fft) +1;
        spec_noise(b) = 0;
    end
    noi = sum(spec_noise(1:floor(N_fft/2/OSR)));
end

% Calculate THD by summing harmonic power
thd = 0;
for i = 2:nTHD
    b = alias(round((bin_r-1)*i),N_fft) +1;
    thd = thd + spec(b);
end

THD = 10*log10(thd/sigs);
SNR = 10*log10(sig/noi);
NF = SNR - pwr;  % Noise floor relative to 0 dBFS

% Finalize plot formatting and annotations
if(dispPlot)
    % Set axis limits based on noise floor
    minx = min(max(median(10*log10(spec_inband))-20,-200),-40);
    axis([Fs/N_fft,Fs/2,minx,0]);
    if(label)
        % Draw signal bandwidth limit
        plot([1,1]*Fs/2/OSR,[0,-1000],'--');
        % Determine text position based on scale and signal location
        if(OSR>1)
            TX = 10^(log10(Fs)*0.01+log10(Fs/N_fft)*0.99);
        else
            if(bin/N_fft < 0.2)
                TX = Fs*0.3;
            else
                TX = Fs*0.01;
            end
        end

        TYD = minx*0.06;  % Text vertical spacing

        % Format sampling frequency with SI prefixes
        if(Fs >= 10^9)
            txt_fs = num2str(Fs/10^9,'%.1fG');
        elseif(Fs >= 10^6)
            txt_fs = num2str(Fs/10^6,'%.1fM');
        elseif(Fs >= 10^3)
            txt_fs = num2str(Fs/10^3,'%.1fK');
        elseif(Fs >= 1)
            txt_fs = num2str(Fs,'%.1f');
        else
            txt_fs = num2str(Fs,'%.3f');
        end

        % Format input frequency with SI prefixes
        Fin = (bin_r-1)/N_fft*Fs;
        if(Fin >= 10^9)
            txt_fin = num2str(Fin/10^9,'%.1fG');
        elseif(Fin >= 10^6)
            txt_fin = num2str(Fin/10^6,'%.1fM');
        elseif(Fin >= 10^3)
            txt_fin = num2str(Fin/10^3,'%.1fK');
        elseif(Fin >= 1)
            txt_fin = num2str(Fin,'%.1f');
        else
            txt_fin = num2str(bin/N_fft*Fs,'%.3f');
        end

        % Display performance metrics
        text(TX,TYD,['Fin/Fs = ',txt_fin,' / ',txt_fs,' Hz']);

        text(TX,TYD*2,['ENoB = ',num2str(ENoB,'%.2f')]);
        text(TX,TYD*3,['SNDR = ',num2str(SNDR,'%.2f'),' dB']);
        text(TX,TYD*4,['SFDR = ',num2str(SFDR,'%.2f'),' dB']);
        text(TX,TYD*5,['THD = ',num2str(THD,'%.2f'),' dB']);
        text(TX,TYD*6,['SNR = ',num2str(SNR,'%.2f'),' dB']);
        text(TX,TYD*7,['Noise Floor = ',num2str(NF,'%.2f'),' dB']);

        % Display additional metrics and noise floor line
        if (OSR>1)
            text(bin/N_fft*Fs,min(pwr,TYD/2),['Sig = ',num2str(pwr,'%.2f'),' dB']);
            semilogx([Fs/N_fft,Fs/2/OSR],-[1,1]*(NF+10*log10(N_fft/2/OSR)),'r--');
            text(TX,TYD*8,['NSD = ',num2str(NF+10*log10(Fs/2/OSR),'%.2f'),' dBFS/Hz']);
            text(TX,TYD*9,['OSR = ',num2str(OSR,'%.2f')]);
        else
            % Position signal power label to avoid signal peak
            if(bin/N_fft>0.4)
                text((bin/N_fft-0.01)*Fs,min(pwr,TYD/2),['Sig = ',num2str(pwr,'%.2f'),' dB'],'horizontalAlignment','right');
            else
                text((bin/N_fft+0.01)*Fs,min(pwr,TYD/2),['Sig = ',num2str(pwr,'%.2f'),' dB']);
            end
            plot([0,Fs/2],-[1,1]*(NF+10*log10(N_fft/2/OSR)),'r--');
            text(TX,TYD*8,['NSD = ',num2str(NF+10*log10(Fs/2/OSR),'%.2f'),' dBFS/Hz']);
        end
    end
    % Set axis labels and title
    xlabel('Freq (Hz)');
    ylabel('dBFS');
    if(N_run > 1)
        if(coAvg)
            title(sprintf('Power Spectrum (%dx Jointed)',N_run));
        else
            title(sprintf('Power Spectrum (%dx Averaged)',N_run));
        end
    else
        title('Power Spectrum');
    end
end

% Assign output variables with new names
enob = (SNDR-1.76)/6.02;
sndr = SNDR;
sfdr = SFDR;
snr = SNR;
thd = THD;
sigpwr = pwr;
noi = NF;
nsd = NF + 10*log10(Fs/2/OSR);

if(~dispPlot)
    h = [];
end

% Nested functions for embedded window generation (no toolbox required)
    function w = rectwin_emb(N)
        % RECTWIN_EMB Embedded rectangle (boxcar) window function
        %   w = RECTWIN_EMB(N) returns an N-point rectangle window in a row vector
        %   This is a simple embedded implementation that doesn't require
        %   the Signal Processing Toolbox
        w = ones(1, N);
    end

    function w = hannwin(N)
        % HANNWIN Embedded Hanning window function
        %   w = HANNWIN(N) returns an N-point Hanning (raised cosine) window
        %   in a row vector. This is a simple embedded implementation that
        %   doesn't require the Signal Processing Toolbox
        %
        %   The Hanning window is defined as:
        %   w(n) = 0.5 * (1 - cos(2*pi*n/(N-1))) for n = 0, 1, ..., N-1
        if N == 1
            w = 1;
        else
            n = 0:(N-1);
            w = 0.5 * (1 - cos(2*pi*n/(N-1)));
        end
    end

end