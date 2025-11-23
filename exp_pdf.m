% ============================================================
% User-defined ADC parameters
% ============================================================
resolution = 12;    % ADC bit resolution
fullscale  = 2;     % normalized data: -1 ~ +1  ==> FS = 2
lsb = fullscale / (2^resolution);


% ============================================================
% Load data
% ============================================================
data = readmatrix(fullfile("ADCToolbox_example_data","sinewave_ref_error_0P020.csv"));


% ============================================================
% Fit sine and compute noise (in raw units)
% ============================================================
[data_fit, freq_est, mag, dc, phi] = sineFit(data);
noise_raw = data - data_fit;

% convert to true LSB units
noise = noise_raw / lsb;    % <<< 关键一步


% ============================================================
% KDE for noise (in LSB)
% ============================================================
[fx, x] = myKDE(noise);


% ============================================================
% Gaussian parameters (in LSB)
% ============================================================
mu  = mean(noise);
sig = std(noise);
var_noise = sig^2;

gauss_pdf = (1/(sig*sqrt(2*pi))) * exp(-(x - mu).^2 / (2*sig^2));


% ============================================================
% Plot
% ============================================================
figure("position",[100,100, 600,400]);
plot(x, fx, 'LineWidth', 1.8); hold on;
plot(x, gauss_pdf, '--r', 'LineWidth', 1.8);

title('Noise Probability Density Function');
xlabel('Noise (LSB)');
ylabel('PDF');
legend('KDE Estimate', 'Gaussian Fit');

fprintf('Noise mean     = %.4e LSB\n', mu);
fprintf('Noise variance = %.4e LSB^2\n', var_noise);
fprintf('Noise std dev  = %.4e LSB\n', sig);



% ============================================================
% KDE function with "minimum ±0.5 LSB range"
% ============================================================
function [f, x_grid] = myKDE(data, varargin)
data = data(:);
N = length(data);

if ~isempty(varargin)
    h = varargin{1};
else
    h = 1.06 * std(data) * N^(-1/5);
end


% Range at least ±0.5 LSB
max_abs_noise = max(abs(data));
xlim_range = max(0.5, max_abs_noise);

x_grid = linspace(-xlim_range, xlim_range, 200);

f = zeros(size(x_grid));
for i = 1:length(x_grid)
    u = (x_grid(i) - data) / h;
    f(i) = mean(exp(-0.5 * u.^2)) / (h * sqrt(2*pi));
end
end
