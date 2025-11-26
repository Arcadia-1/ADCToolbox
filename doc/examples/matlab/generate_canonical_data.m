%% generate_canonical_data.m
% Generates canonical synthetic datasets for documentation examples
%
% Output: CSV files in ../data/
% Each dataset designed to demonstrate specific tool features
%
% Run from: d:\ADCToolbox\doc\examples\matlab\

function generate_canonical_data()

close all; clc;

dataDir = '../data';
if ~isfolder(dataDir)
    mkdir(dataDir);
end

fprintf('=== Generating Canonical Datasets ===\n\n');

%% Dataset 1: Ideal 10-bit Binary-Weighted ADC
fprintf('[1/6] Generating ideal_10bit_sine.csv...\n');
N = 8192;  % 8K samples
freq = 0.1234;  % Normalized frequency (Fin/Fs)
t = (0:N-1)';

% Perfect sinewave: [-1, 1] -> [0, 1023]
signal = sin(2*pi*freq*t);
signal_quantized = round((signal + 1) / 2 * 1023);
signal_quantized = max(0, min(1023, signal_quantized));  % Clip to [0, 1023]

% Convert to 10-bit binary (MSB to LSB)
bits_ideal = zeros(N, 10);
for i = 1:N
    bits_ideal(i, :) = de2bi(signal_quantized(i), 10, 'left-msb');
end

writematrix(bits_ideal, fullfile(dataDir, 'ideal_10bit_sine.csv'));
fprintf('  Samples: %d, Bits: 10, Freq: %.4f\n', N, freq);

%% Dataset 2: 12-bit SAR with Redundancy (15 bits total)
fprintf('[2/6] Generating sar_12bit_redundancy.csv...\n');
N = 16384;
freq = 0.0987;

signal = sin(2*pi*freq*t(1:N));

% Define non-binary weights with 3 redundancy bits
% Bits 1-12: standard, Bits 13-15: redundancy
weights = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, ...
           0.5, 0.25, 0.125] * 2;  % Last 3 are half-weighted redundancy

% Generate 15-bit pattern with these weights
signal_scaled = (signal + 1) / 2 * sum(weights);
bits_redundancy = zeros(N, 15);

residue = signal_scaled;
for i = 1:15
    if residue >= weights(i)
        bits_redundancy(:, i) = 1;
        residue = residue - weights(i);
    else
        bits_redundancy(:, i) = 0;
    end
    % Add small noise to residue for realism
    residue = residue + randn(N,1) * 0.1;
end

writematrix(bits_redundancy, fullfile(dataDir, 'sar_12bit_redundancy.csv'));
fprintf('  Samples: %d, Bits: 15 (12+3 redundancy), Freq: %.4f\n', N, freq);

%% Dataset 3: ADC with INL/DNL (10-bit)
fprintf('[3/6] Generating with_inl_dnl.csv...\n');
N = 8192;
freq = 0.1357;

signal = sin(2*pi*freq*t);
signal_codes = (signal + 1) / 2 * 1023;

% Add synthetic INL (bow shape: 2nd order nonlinearity)
INL = 8 * sin(2*pi * signal_codes / 1024);  % ±8 LSB peak INL
signal_with_inl = signal_codes + INL;

% Add DNL spikes at specific codes
DNL_spikes = zeros(size(signal_with_inl));
DNL_spikes(abs(signal_codes - 256) < 5) = 3;  % Spike at code 256
DNL_spikes(abs(signal_codes - 768) < 5) = -2;  % Dip at code 768
signal_with_inl = signal_with_inl + DNL_spikes;

% Quantize and convert to bits
signal_with_inl = round(signal_with_inl);
signal_with_inl = max(0, min(1023, signal_with_inl));

bits_inl = zeros(N, 10);
for i = 1:N
    bits_inl(i, :) = de2bi(signal_with_inl(i), 10, 'left-msb');
end

writematrix(bits_inl, fullfile(dataDir, 'with_inl_dnl.csv'));
fprintf('  Samples: %d, Bits: 10, Peak INL: ±8 LSB, DNL spikes at codes 256, 768\n', N);

%% Dataset 4: ADC with Harmonic Distortion (10-bit)
fprintf('[4/6] Generating with_harmonic_distortion.csv...\n');
N = 8192;
freq = 0.1111;

% Add 2nd and 3rd harmonics
signal = sin(2*pi*freq*t) + ...
         0.05 * sin(2*pi*freq*2*t) + ...  % HD2: -26 dB
         0.02 * sin(2*pi*freq*3*t);       % HD3: -34 dB

signal_codes = (signal + 1) / 2 * 1023;
signal_codes = round(signal_codes);
signal_codes = max(0, min(1023, signal_codes));

bits_distortion = zeros(N, 10);
for i = 1:N
    bits_distortion(i, :) = de2bi(signal_codes(i), 10, 'left-msb');
end

writematrix(bits_distortion, fullfile(dataDir, 'with_harmonic_distortion.csv'));
fprintf('  Samples: %d, Bits: 10, HD2: -26dB, HD3: -34dB\n', N);

%% Dataset 5: Rank Deficient Bits (for FGCalSine patching demo)
fprintf('[5/6] Generating rank_deficient_bits.csv...\n');
N = 4096;
freq = 0.1523;

signal = sin(2*pi*freq*t(1:N));
signal_codes = round((signal + 1) / 2 * 1023);

% Create 10 bits where bit 3 and 4 are perfectly correlated (rank deficiency)
bits_rank = zeros(N, 10);
for i = 1:N
    bits_rank(i, :) = de2bi(signal_codes(i), 10, 'left-msb');
end

% Make bit 4 = bit 3 (perfect correlation → rank deficiency)
bits_rank(:, 4) = bits_rank(:, 3);

writematrix(bits_rank, fullfile(dataDir, 'rank_deficient_bits.csv'));
fprintf('  Samples: %d, Bits: 10, Bit 3 == Bit 4 (rank deficient)\n', N);

%% Dataset 6: SAR with Overflow (for overflowChk demo)
fprintf('[6/6] Generating sar_with_overflow.csv...\n');
N = 8192;
freq = 0.0876;

signal = sin(2*pi*freq*t);

% Intentionally create overflow by undersized MSB weight
weights_overflow = [400, 256, 128, 64, 32, 16, 8, 4, 2, 1];  % MSB too small!

signal_scaled = (signal + 1) / 2 * sum(weights_overflow);
bits_overflow = zeros(N, 10);

residue = signal_scaled;
for i = 1:10
    if residue >= weights_overflow(i)
        bits_overflow(:, i) = 1;
        residue = residue - weights_overflow(i);
    else
        bits_overflow(:, i) = 0;
    end
end

writematrix(bits_overflow, fullfile(dataDir, 'sar_with_overflow.csv'));
fprintf('  Samples: %d, Bits: 10, MSB weight: 400 (should be 512) → overflow\n', N);

%% Generate README.md
fprintf('\nGenerating data/README.md...\n');
fid = fopen(fullfile(dataDir, 'README.md'), 'w');
fprintf(fid, '# Canonical Datasets for Documentation\n\n');
fprintf(fid, 'These datasets are synthetic ADC outputs designed to demonstrate specific tool features.\n\n');
fprintf(fid, '## Datasets\n\n');
fprintf(fid, '| File | Description | Samples | Bits | Key Features |\n');
fprintf(fid, '|------|-------------|---------|------|-------------|\n');
fprintf(fid, '| `ideal_10bit_sine.csv` | Perfect binary-weighted ADC | 8192 | 10 | Baseline reference, no errors |\n');
fprintf(fid, '| `sar_12bit_redundancy.csv` | SAR with redundancy | 16384 | 15 | 12-bit + 3 redundancy bits |\n');
fprintf(fid, '| `with_inl_dnl.csv` | ADC with nonlinearity | 8192 | 10 | INL bow (±8 LSB), DNL spikes |\n');
fprintf(fid, '| `with_harmonic_distortion.csv` | Harmonic distortion | 8192 | 10 | HD2 (-26dB), HD3 (-34dB) |\n');
fprintf(fid, '| `rank_deficient_bits.csv` | Correlated bits | 4096 | 10 | Bit 3 == Bit 4 (rank deficiency) |\n');
fprintf(fid, '| `sar_with_overflow.csv` | Overflow example | 8192 | 10 | Undersized MSB → overflow |\n');
fprintf(fid, '\n## Usage\n\n');
fprintf(fid, 'These datasets are used by:\n');
fprintf(fid, '- `doc/examples/matlab/generate_all_figures.m` (MATLAB)\n');
fprintf(fid, '- `doc/examples/python/generate_all_figures.py` (Python - future)\n');
fprintf(fid, '\n## File Format\n\n');
fprintf(fid, '- CSV format (no headers)\n');
fprintf(fid, '- Each row: one sample\n');
fprintf(fid, '- Each column: one bit (MSB to LSB, left to right)\n');
fprintf(fid, '- Values: 0 or 1\n');
fclose(fid);

fprintf('\n=== All Canonical Datasets Generated ===\n');
fprintf('Location: %s\n', dataDir);

end
