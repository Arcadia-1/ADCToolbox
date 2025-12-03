%% Quantization Verification for plotspec
clear;
close all;
clc;

N = 2^13;
J = 323;
sig = 0.499 * sin((0:N - 1)'*J*2*pi/N) + 0.5;

nbits = 1;
sig_quantized = floor(sig*2^nbits) / 2^nbits;
figure('Visible', 'off');
enob_new = plotspec(sig_quantized, 'label', 0, 'harmonic', 5);
enob_old = specPlot(sig_quantized, 'label', 0, 'harmonic', 5);
close(gcf);

fprintf('[%2d bit] ENOB (new) = %.4f, ENOB (old) = %.4f\n',nbits, enob_new, enob_old);

%% Sweep 1 to 40 bits
fprintf('[Sweeping 1-40 bits]\n');
bit_sweep = 1:20;
enob_new_results = zeros(size(bit_sweep));
enob_old_results = zeros(size(bit_sweep));
sndr_results = zeros(size(bit_sweep));

for idx = 1:length(bit_sweep)
    nbits = bit_sweep(idx);
    sig_quantized = floor(sig*2^nbits) / 2^nbits;

    [enob_new, sndr] = plotspec(sig_quantized, 'label', 0, 'harmonic', 5, 'isplot', 0);
    enob_old = specPlot(sig_quantized, 'label', 0, 'harmonic', 5, 'isplot', 0);

    enob_new_results(idx) = enob_new;
    enob_old_results(idx) = enob_old;
    sndr_results(idx) = sndr;

    if mod(nbits, 5) == 0
        fprintf('[%2dbit] ENOB_new=%.2f, ENOB_old=%.2f, SNDR=%.2f\n', nbits, enob_new, enob_old, sndr);
    end
end
fprintf('\n');

%% Plot Results
figure('Position', [100, 100, 800, 600]);

plot(bit_sweep, bit_sweep, 'k--', 'LineWidth', 1.5); hold on;
plot(bit_sweep, enob_old_results, 'r-o', 'LineWidth', 2, 'MarkerSize', 4);
plot(bit_sweep, enob_new_results, 'b-o', 'LineWidth', 2, 'MarkerSize', 4);
grid on;
xlabel('Quantization Bits', 'FontSize', 14);
ylabel('ENOB (bits)', 'FontSize', 14);
title('ENOB vs Quantization Bits', 'FontSize', 16);
legend('Reslotion', 'specPlot (old)', 'plotspec (new)', 'Location', 'northwest');
xlim([1, max(bit_sweep)]);
ylim([1, max(bit_sweep)]);
set(gca, 'FontSize', 12);