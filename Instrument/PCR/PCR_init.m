function [h] = PCR_init(addr,ch341)

    if(nargin < 2)
        ch341 = 0;
    end
    
    addr_disp = strrep(addr,'z','Z');

    addr = strrep(addr,'2',' ');
    addr = strrep(addr,'Z','2');
    addr = strrep(addr,'z','2');
    
    if(~isempty(regexp(addr,'[^012]')) || length(addr) ~= 2)
        error('PCR: Invalid addr');
    end
    
    abs_addr = (sum((addr-'0').*[32,2]) + 42)*2;

    h = struct('ch341',ch341, 'type', 'PCR', 'addr',abs_addr, 'addr_disp', addr_disp);

    fprintf('PCR: Reseting PCR@%s ',addr_disp);
    CH341_I2C(abs_addr, [hex2dec('FD')], 0); % reset
    
    for i1=1:10
        fprintf('.');
        if(PCR_checkConn(h))
            break
        end
    end
    PCR_checkConn(h);
    fprintf(' Success!\n');
    
    PCR_confDefault(h);

end