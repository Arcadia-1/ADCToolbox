function bin = findBin(Fs,Fin,N)
    bin = floor(Fin/Fs*N);
    while(~isprime(bin))
        bin = bin+1;
    end
end