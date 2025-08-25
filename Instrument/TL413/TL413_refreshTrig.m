function TL413_refreshTrig()

% Do not call this manually

    global TL413_iTrig
    global TL413_iFreq
    global TL413_iExtClk
    global TL413_iPassCount
    global TL413_iTrPos

    TL413_iTrig.iFreq = TL413_iFreq;
    TL413_iTrig.iExtClk = TL413_iExtClk;
    TL413_iTrig.iPassCount = TL413_iPassCount;
    TL413_iTrig.iTrPos = TL413_iTrPos;
        
end