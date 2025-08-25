% handle = visa('ni','USB0::0x05E6::0x2450::04507202::INSTR');
% fopen(handle);


res = zeros([1,1000]);

hbar = waitbar(0,'');
for ii = 1:100
    fprintf(handle,':READ?');
    res(ii) = fscanf(handle,'%g');
    waitbar(ii/100,hbar,'');
end

mean(res)

% fclose(handle);