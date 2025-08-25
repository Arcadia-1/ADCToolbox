function TL413_setMode(mode)

% Mode = 
%     0: Synchronous Sampling, 32CH
%        Asynchronous Mode Below
%     1:    200M Max, 32CH
%     2:    200M Max, 16CH
%     3:    200M Max, 8CH
%     4:    200M Max, 4CH
%     5:    250M Max, 32CH
%     6:    500M Max, 16CH
%     7:    1G Max, 8CH
%     8:    2G Max, 4CH

    if(mode < 0 || mode > 8)
        error('TL413: Mode setting out of range (0-8)');
    end
    if(mode == 0)
        hwCode = 1320;
    else
        hwCode = 1299+mode;
    end
    
    modeName = {
    'TLHW_SYNC_32CH',...
    'TLHW_200M_32CH',...
    'TLHW_200M_16CH',...
    'TLHW_200M_8CH',...
    'TLHW_200M_4CH',...
    'TLHW_250M_32CH',...
    'TLHW_500M_16CH',...
    'TLHW_1_0G_8CH',...
    'TLHW_2_0G_4CH'};

    TL413_check();
    
    fprintf('TL413: Setting mode to %s ... ',modeName{mode+1});
    iHwMode = libpointer('int32Ptr', hwCode);
    calllib('TLSDK64', 'ulaSDKSetHwInfo', 5, iHwMode);
    clear iHwMode
    
    TL413_check();
    
    fprintf('OK!\n');
    
end