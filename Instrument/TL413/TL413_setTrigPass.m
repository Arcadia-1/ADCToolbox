function TL413_setTrigPass(num)

    global TL413_iPassCount
    
    if(num < 0)
        error('TL413: Trig pass count setting out of range (>0)');
    end
    
    fprintf('TL413: Set trig pass count to %d\n',num);
    TL413_iPassCount = num;

end