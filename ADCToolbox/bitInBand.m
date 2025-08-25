function [dout] = bitInBand(din,bands)

    [N,M] = size(din);
    if(N < M)
        bits = bits';
        [N,M] = size(bits);
    end

    [P,Q] = size(bands);
    if(Q ~= 2)
        error('bands must be with 2 columns');
    end

    spec = fft(din);

    mask = zeros(N,1);
    
    for i1 = 1:P
        n1 = round(min(bands(i1,:))*N);
        n2 = round(max(bands(i1,:))*N);
        ids = alias(n1:n2,N);
        mask(ids+1) = 1;
        mask(end-ids+1) = 1;
    end

    spec = spec.*(mask(1:N,1)*ones(1,M));

    dout = real(ifft(spec));

end