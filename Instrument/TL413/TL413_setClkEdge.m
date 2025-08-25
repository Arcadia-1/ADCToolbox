function TL413_setClkEdge(edge)

% edge = 0 : rising edge of CLK0
% edge = 1 : falling edge of CLK0
% edge = 2 : both edges of CLK0

    global TL413_iExtClk
    
    if(edge < 0 || edge > 2)
        error('TL413: Clk edge setting out of range (0/1/2)');
    end
    
    clkEdgeName = {'Rising Edge','Falling Edge','Both Edges'};
    
    fprintf('TL413: Sample on %s of CLK0\n',clkEdgeName{edge+1});
    TL413_iExtClk = edge;

end