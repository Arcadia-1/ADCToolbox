function TL413_setSamRate(fs)

    global TL413_iFreq
    
    if(fs < 1 || fs > 2*10^8)
        error('TL413: Sampling rate setting out of range (1Hz~2GHz)');
    end
    
    fprintf('TL413: Set sampling rate to %d MHz\n',floor(fs/10^6));
    TL413_iFreq = fs;

end