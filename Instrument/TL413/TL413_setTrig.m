function TL413_setTrig(lvl, ch, type, logic)
    
    global TL413_iTrig
    
    TL413_check();
    
    if(lvl < 0 || lvl > 15)
        error('TL413: Trigger condition level setting out of range (0~15)');
    elseif(ch < 0 || ch > 31)
        error('TL413: Trigger channel id setting out of range (0~31)');
    elseif(type < 0 || type > 5)
        error('TL413: Trigger type setting out of range (0~5)');
    end
    if(nargin < 4)
        logic = 2;
    elseif(logic < 0 || logic > 2)
        error('TL413: Trigger logic setting out of range (0/1/2)');
    end
    
    typeCode = [8,6,0,4,2,10];
    typeName = {'Level-High', 'Level-Low', 'Rising-Edge', 'Falling-Edge', 'Any-Edge'};
    
    if(type == 0)
        fprintf('TL413: Setting Lv.%d trigger to [No Condition] ... ',lvl);
    else
        fprintf('TL413: Setting Lv.%d trigger to [%s on Ch.%d] ... ',lvl,typeName{type},ch);
    end
    
    TL413_refreshTrig;
    iTrigPtr = libpointer('SDKTRIG', TL413_iTrig);
    calllib('TLSDK64', 'ulaSDKSetChTrigger', iTrigPtr, lvl, ch, typeCode(type+1), logic);
    TL413_iTrig = iTrigPtr.value;
    clear iTrigPtr
    
    TL413_check();
   
    fprintf('OK!\n');
    
end