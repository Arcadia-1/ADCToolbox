function TL413_setTrigPos(num)

    global TL413_iTrPos
    
    if(num < 1)
        error('TL413: Trig position setting out of range (>1)');
    end
    
    fprintf('TL413: Set trig position to %d\n',num);
    TL413_iTrPos = num;

end