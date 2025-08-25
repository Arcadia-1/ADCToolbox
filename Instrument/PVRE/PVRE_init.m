function [h1, h2] = PVRE_init(addr,ch341)

    if(nargin < 2)
        ch341 = 0;
    end
    
    addr_disp = strrep(addr,'z','Z');

    addr = strrep(addr,'2',' ');
    addr = strrep(addr,'1','2');
    addr = strrep(addr,'Z','1');
    addr = strrep(addr,'z','1');
    
    if(~isempty(regexp(addr,'[^012]')) || length(addr) ~= 2)
        error('PVRE: Invalid addr');
    end
    
    abs_addr = (sum((addr-'0').*[3,1]) + hex2dec('6C'))*2;
    
    h1 = struct('ch341',ch341, 'type', 'PVRE', 'addr', abs_addr, 'ch', 0, 'addr_disp', addr_disp);
    h2 = struct('ch341',ch341, 'type', 'PVRE', 'addr', abs_addr, 'ch', 1, 'addr_disp', addr_disp);
   
    
    fprintf('PVRE: Reseting PVRE@%s ',addr_disp);
    CH341_I2C(abs_addr, [hex2dec('16')], 0); % restore E2PROM (reset)
    
    for i1=1:10
        fprintf('.');
        if(PVRE_checkConn(h1))
            break
        end
    end
    PVRE_checkConn(h1);
    fprintf(' Success!\n');
    
    PVRE_confDefault(h1);
    PVRE_confDefault(h2);
    
    PVRE_setVout(h1, 0);
    PVRE_setVout(h2, 0);
    
    CH341_I2C(h1.addr, [hex2dec('01'),hex2dec('80')], 0);
    CH341_I2C(h2.addr, [hex2dec('01'),hex2dec('80')], 0);
end