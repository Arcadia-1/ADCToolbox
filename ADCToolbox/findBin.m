function bin = findBin(Fs,Fin,N)
    bin = floor(Fin/Fs*N);
    while(gcd(bin,N)>1)
        bin = bin+1;
    end
end