function [noise_lsb, mu, sigma, KL_divergence, x, fx, gauss_pdf] = errPDF(err_data, varargin)

p = inputParser;
addParameter(p, "Resolution", 12);
addParameter(p, "FullScale",  1);
parse(p, varargin{:});
resolution = p.Results.Resolution;
fullscale  = p.Results.FullScale;

lsb = fullscale / (2^resolution);
noise_lsb = err_data(:) / lsb;
n = noise_lsb;
N = length(n);

h = 1.06 * std(n) * N^(-1/5);
max_abs_noise = max(abs(n));
xlim_range = max(0.5, max_abs_noise);
x = linspace(-xlim_range, xlim_range, 200);
fx = zeros(size(x));

for i = 1:length(x)
    u = (x(i) - n) / h;
    fx(i) = mean(exp(-0.5 * u.^2)) / (h * sqrt(2*pi));
end

mu = mean(n);
sigma = std(n);
gauss_pdf = (1/(sigma*sqrt(2*pi))) * exp(-(x - mu).^2 / (2*sigma^2));


dx = x(2) - x(1);
p = fx + eps;
q = gauss_pdf + eps;
KL_divergence = sum(p .* log(p ./ q)) * dx;



hold on;
plot(x, fx, 'LineWidth', 2, 'DisplayName',"KDE Estimate");
label = sprintf("Gaussian Fit (KL = %0.4f, \\mu=%0.2f, \\sigma=%0.2f)", KL_divergence, mu, sigma);
plot(x, gauss_pdf, '--r', 'LineWidth', 2, 'DisplayName',label);
xlabel("Noise (LSB)");
ylabel("PDF");
legend("Location","northwest");
grid on;
y_max = max([fx, gauss_pdf])*1.3; 
ylim([0,y_max])
set(gca,'FontSize',14)
end
