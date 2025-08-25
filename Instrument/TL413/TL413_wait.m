function overtime = TL413_wait(max_time)

    if(nargin == 0)
        max_time = 0;
    elseif(nargin < 0)
        error('TL413: Wait time must be non-negative');
    end
    
    if(calllib('TLSDK64', 'ulaSDKIsCaptureReady'))
        overtime = 0;
        return;
    end
    
    if(max_time == 0)
        fprintf('TL413: Waiting capture ... ');
    else
        fprintf('TL413: Waiting capture (%.1fsec max)... ',max_time);
    end
    
    tic;
    while(1)
        if(calllib('TLSDK64', 'ulaSDKIsCaptureReady'))
            overtime = 0;
            break;
        end
        TL413_check();
        if(max_time > 0 && toc > max_time)
            overtime = 1;
            break;
        end
    end
    
    if(overtime)
        fprintf('Overtime!\n');
    else
        fprintf('Data Ready!\n');
    end

end