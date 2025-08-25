function [weight, offset, postCal, ideal, error] = FGCalSineOS(bits, rel_freq, osr, order, nomWeight)

% foreground calibration by inputting sinewave
% considering oversampling - only inband error is considered
%
% data format: MSB left - LSB right
% bits - the raw data, N row by M col, N is data points, M is bitwidth
% rel_freq - the reletive test freq Fin/Fs
% osr - the oversampling rate
% order - the order of distortion excluded
% nomWeight is the nominal weight (no normalization required), this is
%   optional for enhanced solving robustness
%
% weight is normalized by MSB. i.e., weigth[MSB] = 1
% offset is normalized by MSB too
% postCal is the signal after calibration
% ideal is the ideal sine signal
% error is the residue errors after calibration


[N, M] = size(bits);
N2 = floor(N/2/osr);

bit_spec = fft(bits)/N;
bit_spec = bit_spec(1:N2, :);

if (nargin < 4)
    order = 1;
end
order = max(order, 1);

if (nargin >= 5)
    dgain = 1 ./ sqrt(nomWeight);
    dither = rand([N2, M - 1]) .* (ones([N2, 1]) * dgain(2:end));
    bits_patch = bit_spec + [-dither * nomWeight(2:end)' / nomWeight(1), dither];
else
    bits_patch = bit_spec;
end

xct = cos(rel_freq*(0:(N-1))*2*pi);
xst = sin(rel_freq*(0:(N-1))*2*pi);

tone = zeros([N2,order]);
for ii = 1:order
    b = round(ii*rel_freq*N)+1;
    if(b <= N2)
        tone(b, ii) = 1;
    end
end

A = [bits_patch(1:N2, 2:M), [1;zeros(N2-1,1)], tone];

b = -bits_patch(1:N2, 1);

x = linsolve(A, b);

weight = [1, real(x(1:M-1)')];

offset = -real(x(M));

postCal = weight * bits' - offset;

ideal = (-xct * real(x(M+1)) + xst * imag(x(M+1)))*2;
    
error = postCal - ideal;

end