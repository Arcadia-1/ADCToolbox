clear; close all; clc;

subFolder = fullfile("dataset", "dout");
if ~exist(subFolder, 'dir'), mkdir(subFolder); end

%%
N_STAGES = 8;
N_BITS_PER_STAGE = 3; % N_i = 3 bits for stages 1 to 7
GAIN_PER_STAGE = 4; % G_i = 4 for stages 1 to 7
N_LAST_STAGE = 4; % N_8 = 4 bits for the final stage

N = 2^13;
J = findBin(1, 0.0789, N);
A = 0.49; % Full scale amplitude
sig = A * sin((0:N - 1)'*J*2*pi/N) + randn(N, 1)*1e-6 + 0.5; % Base sinewave (zero mean)

N_i = N_BITS_PER_STAGE;
G_i = GAIN_PER_STAGE;
total_bits = 0;
digital_code_cells = cell(1, N_STAGES); % Cell array to store bits from each stage
weights_cells = cell(1, N_STAGES);
redundancy = log2(2^N_i/G_i); % Redundancy per stage (1 bit)

current_signal = sig;

% --- Pipeline Stage Loop (Stages 1 to 7) ---
for i = 1:(N_STAGES - 1) % Loop for stages 1 through 7

    % 1. Calculate ideal offset for residue centering
    offset_i = (1 - G_i * 1 / 2^N_i) / 2;

    % 2. Quantize current signal (MSB for this stage)
    quantized_i = floor(current_signal*2^N_i) / 2^N_i;

    % 3. Calculate residue
    residue_i = current_signal - quantized_i;

    % 4. Generate next stage input: amplify residue and add offset
    next_stage_input = G_i * residue_i + offset_i;

    % 5. Store digital code for this stage
    digital_code_cells{i} = binSaturate(quantized_i*2^N_i, N_i);
    total_bits = total_bits + N_i;

    % 6. Update current_signal for next iteration
    current_signal = next_stage_input;

    % 7. Calculate weights
    % effective_bits_remaining: Total effective bits *from stage i onwards*
    effective_bits_remaining = N_i * (N_STAGES - i) - redundancy * (N_STAGES - i - 1) + (N_LAST_STAGE - N_i);
    weights_cells{i} = 2.^(effective_bits_remaining - 1:-1:effective_bits_remaining - N_i);
end

% --- Last Stage (Stage 8: Final Quantization) ---
last_stage_index = N_STAGES;

% Final Quantization (G=1 implicitly, no further residue subtraction)
quantized_last = floor(current_signal*2^N_LAST_STAGE) / 2^N_LAST_STAGE;

% Store digital code and update total bits
digital_code_cells{last_stage_index} = binSaturate(quantized_last*2^N_LAST_STAGE, N_LAST_STAGE);
total_bits = total_bits + N_LAST_STAGE;

% Weights for the last stage (fully quantizes remaining residue)
weights_cells{last_stage_index} = 2.^((N_LAST_STAGE - 1):-1:0);


% --- Final Digital Code and Weight Combination ---
dout = cell2mat(digital_code_cells);
weights = cell2mat(weights_cells);
weights = weights / sum(weights); % Normalize weights

ENoB1 = specPlot(digital_code_cells{1}*weights_cells{1}', "isplot", 0);
ENoB = specPlot(dout*weights', "isplot", 0);
fprintf("[%s] [ENoB1 = %0.2f bits] [ENoB2 = %0.2f bits]\n", mfilename, ENoB1, ENoB);


total_redundancy = redundancy * (N_STAGES - 1); % Redundancy only applied in stages 1 to 7
effective_res = total_bits - total_redundancy;


figure
overflowChk(dout, weights);

title_str = sprintf('8-stage Pipeline ADC (N_i=%d, G_i=%d, N_8=%d, Resolution=%.1f bits)', ...
    N_BITS_PER_STAGE, GAIN_PER_STAGE, N_LAST_STAGE, effective_res);
title(title_str);

filename = fullfile("dataset/dout", sprintf("dout_Pipeline_3bx4x%d_4b.csv", N_STAGES));
fprintf("[Save data into file] -> [%s]\n", filename);
writematrix(dout, filename);

function bits = binSaturate(x, N)
% Encode bits: Prevent digital overflow/wrap-around; saturate codes exceeding the maximum.
x = floor(x); % ensure integer
x = min(max(x, 0), 2^N-1); % saturate to [0, 2^N - 1]
bits = dec2bin(x, N) - '0'; % fixed width N bits
end
