function [acf, lags] = errAutoCorrelation(err_data, varargin)
% errACF - Compute and plot autocorrelation of error signal.
% The function plots into current axes and returns the ACF and lags.
%
% Usage:
%   [acf, lags] = errACF(err_data)
%   [...] = errACF(err_data, "MaxLag", 200)
%   [...] = errACF(err_data, "Normalize", true)

% Parse parameters
p = inputParser;
addParameter(p, "MaxLag", 100);       % number of lags to compute
addParameter(p, "Normalize", true);   % normalize acf so acf(0) = 1
parse(p, varargin{:});

maxLag    = p.Results.MaxLag;
normalize = p.Results.Normalize;

% Ensure column data
e = err_data(:);
N = length(e);

% Subtract mean (ACF doesn't require mean unless you want to see bias)
e = e - mean(e);

% Preallocate
lags = -maxLag:maxLag;
acf  = zeros(size(lags));

% Compute autocorrelation manually (consistent with toolbox style)
for k = 1:length(lags)
    lag = lags(k);
    if lag >= 0
        x1 = e(1:N-lag);
        x2 = e(1+lag:N);
    else
        lag2 = -lag;
        x1 = e(1+lag2:N);
        x2 = e(1:N-lag2);
    end
    acf(k) = mean(x1 .* x2);
end

% Normalize if required
if normalize
    acf = acf / acf(lags==0);
end


% Plot
plot(lags, acf, 'LineWidth', 2);
grid on;
xlabel("Lag (samples)");
ylabel("Autocorrelation");

set(gca, "fontsize", 16)