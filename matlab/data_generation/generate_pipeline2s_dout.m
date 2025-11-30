clear;
close all;
clc;

N = 2^13;
J = findBin(1, 0.0789, N);
A = 0.49;
sig = A * sin((0:N - 1)*J*2*pi/N) + 0.5; % Base sinewave (zero mean)

% stage parameters
N1 = 3;
G1 = 8;
N2 = 8;

offset1 = (1 - G1 * 1 / 2^N1) / 2;

msb = floor(sig*2^N1) / 2^N1; % stage 1
residue1 = sig - msb;

lsb = floor((G1 * residue1 + offset1)*2^N2) / 2^N2; % stage 2

msb_bits = binSaturate(msb*2^N1, N1);
lsb_bits = binSaturate(lsb*2^N2, N2);

dout = [msb_bits, lsb_bits];


R1 = log2(2^N1/G1); % redundancy of stage 1

% bit weights
w1 = 2.^((N1 + N2 - R1 - 1):-1:(N2 - R1));
w2 = 2.^((N2 - 1):-1:0);

weights = [w1, w2];
weights = weights / sum(weights);

ENoB1 = specPlot(msb_bits*w1', "isplot", 0);
ENoB = specPlot(dout*weights', "isplot", 0);
fprintf("[%s] [ENoB1 = %0.2f bits] [ENoB2 = %0.2f bits]\n", mfilename, ENoB1, ENoB);

figure;
overflowChk(dout, weights);
title_str = sprintf('2-stage Pipeline (N1=%d, G1=%d, N2=%d, G2=%d, N3=%d)', N1, G1, N2);
title(title_str);

filename = fullfile("dataset/dout", ...
    sprintf("dout_Pipeline_%dbx%d_%db.csv", N1, G1, N2));
fprintf("[Save data into file] -> [%s]\n", filename);
writematrix(dout, filename);

function bits = binSaturate(x, N)
% Encode bits: Prevent digital overflow/wrap-around; saturate codes exceeding the maximum.
x = floor(x); % ensure integer
x = min(max(x, 0), 2^N-1); % saturate to [0, 2^N - 1]
bits = dec2bin(x, N) - '0'; % fixed width N bits
end
