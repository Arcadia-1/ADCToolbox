function TL413_setVTh(vth,pod)

% Pod 0 : CH0 - CH7
% Pod 1 : CH8 - CH15
% Pod 2 : CH16 - CH23
% Pod 3 : CH24 - CH31

    if(nargin<2)
        pod = -1;
    elseif(pod<0 || pod > 3)
        error('TL413: Threshold POD out of range (0~3)');
    end
    if(vth < -5 || vth > 5)
        error('TL413: Threshold setting out of range (-5V~5V)');
    end
    
    TL413_check();
    
    vth_mv = round(vth*1000);
    if(pod == -1)
        fprintf('TL413: Setting threshold voltage of all POD to %dmV ... ',vth_mv);
        calllib('TLSDK64', 'ulaSDKThreshold', 0, vth_mv);
        calllib('TLSDK64', 'ulaSDKThreshold', 1, vth_mv);
        calllib('TLSDK64', 'ulaSDKThreshold', 2, vth_mv);
        calllib('TLSDK64', 'ulaSDKThreshold', 3, vth_mv);
    else
        fprintf('TL413: Setting threshold voltage of POD-%d to %dmV ... ',pod,vth_mv);
        calllib('TLSDK64', 'ulaSDKThreshold', pod, vth_mv);
    end
    
    TL413_check();
    
    fprintf('OK!\n');

end