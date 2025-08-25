weight = 2.^[18:-1:15,15:-1:12,12:-1:8,8,7,6,6,5,4,4,3,2,1,1];
mismatch = 0.05;

weight_real = weight.*(1+mismatch./sqrt(weight));

% cal --
% preset
% convert
% solve

% test --
% sine
% convert
% fft