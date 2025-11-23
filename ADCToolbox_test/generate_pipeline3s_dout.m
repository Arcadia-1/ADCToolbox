clear;
close all;
clc;

N = 2^16;
J = findBin(1, 0.0789, N);
A = 1;
sig = A * sin((0:N - 1)*J*2*pi/N); % Base sinewave (zero mean)

% stage parameters
N1 = 4;
G1 = 8;
N2 = 4;
G2 = 8;
N3 = 8;

offset1 = (1 - G1 * 1 / 2^N1) / 2;
offset2 = (1 - G2 * 1 / 2^N2) / 2;

msb = floor(sig*2^N1) / 2^N1; % stage 1
residue1 = sig - msb;

lsb = floor((G1 * residue1 + offset1)*2^N2) / 2^N2; % stage 2
residue2 = G1 * residue1 + offset1 - lsb;

lsb2 = floor((G2 * residue2 + offset2)*2^N3) / 2^N3; % stage 3

msb_bits = binSaturate(msb*2^N1, N1);
lsb_bits = binSaturate(lsb*2^N2, N2);
lsb2_bits = binSaturate(lsb2*2^N3, N3);

dout = [msb_bits, lsb_bits, lsb2_bits];


R1 = log2(2^N1/G1); % redundancy of stage 1
R2 = log2(2^N2/G2); % redundancy of stage 2

% bit weights
w1 = 2.^((N1 + N2 + N3 - R2 - R1 - 1):-1:(N2 + N3 - R2 - R1));
w2 = 2.^((N2 + N3 - R2 - 1):-1:N3 - R2);
w3 = 2.^((N3 - 1):-1:0);

weights = [w1, w2, w3];
weights = weights / sum(weights);

figure;
overflowChk(dout, weights);
title_str = sprintf('3-stage Pipeline (N1=%d, G1=%d, N2=%d, G2=%d, N3=%d)', N1, G1, N2, G2, N3);
title(title_str);

filename = fullfile("ADCToolbox_example_data", ...
    sprintf("dout_Pipeline_%dbx%d_%dbx%d_%db.csv", N1, G1, N2, G2, N3));
fprintf("[Save data into file] -> [%s]\n", filename);
writematrix(dout, filename);

function bits = binSaturate(x, N)
% Encode bits: Prevent digital overflow/wrap-around; saturate codes exceeding the maximum.
x = floor(x); % ensure integer
x = min(max(x, 0), 2^N-1); % saturate to [0, 2^N - 1]
bits = dec2bin(x, N) - '0'; % fixed width N bits
end
