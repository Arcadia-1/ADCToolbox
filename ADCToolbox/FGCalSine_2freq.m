function [weight,offset,postCal1,postCal2] = FGCalSine_2freq(bits1,bits2,rel_freq1,rel_freq2,order,nomWeight)

% foreground calibration by inputting sinewave
% two-frequency version - calibrate under two different frequencies, to
% exclude frequency dependent error
%
% data format: MSB left - LSB right
% bits is the raw data, N row by M col, N is data points, M is bitwidth
% rel_freq is the reletive test freq Fin/Fs
% order is the order of distortion excluded
%
% weight is normalized by MSB. i.e., weigth[MSB] = 1
% offset is normalized by MSB too
% postCal is the signal after calibration


    [N1,M1] = size(bits1);
    [N2,M2] = size(bits2);
    
    if(M1 ~= M2)
        error('Data bit width mismatch');
    end
    M = M1;
    
    if(nargin < 3)
        order = 1;
    end
    order = max(order,1);
    
    if(nargin >= 4)
        dgain = 1./sqrt(nomWeight);
        dither = rand([N,M-1]).*(ones([N,1])*dgain(2:end));
        bits_patch1 = bits1 + [-dither*nomWeight(2:end)'/nomWeight(1),dither];   
        bits_patch2 = bits2 + [-dither*nomWeight(2:end)'/nomWeight(1),dither];   
    else
        bits_patch1 = bits1;
        bits_patch2 = bits2;
    end

    xc1 = cos(alias(rel_freq1*(1:order)',1)*(0:(N1-1))*2*pi);
    xs1 = sin(alias(rel_freq1*(1:order)',1)*(0:(N1-1))*2*pi);
    xc2 = cos(alias(rel_freq1*(1:order)',1)*(0:(N2-1))*2*pi);
    xs2 = sin(alias(rel_freq1*(1:order)',1)*(0:(N2-1))*2*pi);
    
    A = [bits_patch1(1:N1,2:M),ones(N1,1),xc1',xs1'; bits_patch2(1:N2,2:M),ones(N2,1),xc2',xs2'];

    b = [-bits_patch1(1:N1,1); -bits_patch2(1:N2,1)];
    
    x = linsolve(A,b);
    
    weight = [1,x(1:M-1)'];
    
    offset = -x(M);
    
    postCal1 = weight*bits1' - offset;
    postCal2 = weight*bits2' - offset;

end