function [coarse,fine,sam] = data_parse(data,n_fft)

    data = data(end-n_fft*32+1:end);
    data = reshape(data',[32,n_fft])';
    
    coarse = [0,0,2.^(13:-1:0),zeros([1,16])]*data';
    fine = [zeros([1,16]),0,0,2.^(13:-1:0)]*data';
    sam = [8,4,zeros([1,14]),2,1,zeros([1,14])]*data';
    
end