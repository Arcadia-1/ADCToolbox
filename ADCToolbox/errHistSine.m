function [emean, erms, phase_code, edata] = errHistSine(data, varargin)

    % mode = 0 : phase as x axis; 
    % mode >= 1 : code as axis;

    p = inputParser;
    addOptional(p, 'bin', 100, @(x) isnumeric(x) && isscalar(x) && (x > 0));
    addOptional(p, 'fin', 0, @(x) isnumeric(x) && isscalar(x) && (x > 0) && (x < 1));
    addOptional(p, 'disp', 1);
    addOptional(p, 'mode', 0, @(x) isnumeric(x));
    addParameter(p, 'erange', []);
    parse(p, varargin{:});
    bin = round(p.Results.bin);
    fin = p.Results.fin;
    disp = p.Results.disp;
    codeMode = p.Results.mode;
    erange = p.Results.erange;

    if(fin == 0)
        [data_fit,fin,~,~,phi] = sineFit(data);
    else
        [data_fit,~,~,~,phi] = sineFit(data,fin);
    end

    error = data_fit-data;
    
    if(codeMode)
        dat_min = min(data);
        dat_max = max(data);
        bin_wid = (dat_max-dat_min)/bin;
        phase_code =  min(data) + [1:bin]*bin_wid - bin_wid/2;
        
        enum = zeros([1,bin]);
        esum = zeros([1,bin]);
        erms = zeros([1,bin]);
    
        for ii = 1:length(data)
            b = min(floor((data(ii)-dat_min)/bin_wid)+1,bin);
            esum(b) = esum(b) + error(ii);
            enum(b) = enum(b) + 1;
        end
        emean = esum./enum;
        for ii = 1:length(data)
            b = min(floor((data(ii)-dat_min)/bin_wid)+1,bin);
            erms(b) = erms(b) + (error(ii) - emean(b))^2;
        end
        erms = sqrt(erms./enum);

        if(~isempty(erange))
            eid = (data >= erange(1)) & (data <= erange(2));
            edata = error(eid);
        end

        if(disp)
            subplot(2,1,1);
    
            plot(data,error,'r.');
            hold on;
            plot(phase_code,emean,'b-');
            axis([dat_min,dat_max,min(error),max(error)]);
            ylabel('error');
            xlabel('code');

            if(~isempty(erange))
                plot(data(eid),error(eid),'m.');
            end
    
            subplot(2,1,2);
            bar(phase_code,erms);
            axis([dat_min,dat_max,0,max(erms)*1.1]);
            xlabel('code');
            ylabel('RMS error');
        end

    else
        phi_list = mod(phi/pi*180 + (0:length(data)-1)*fin*360,360);
        phase_code = (0:bin-1)/bin*360;

        enum = zeros([1,bin]);
        esum = zeros([1,bin]);
        erms = zeros([1,bin]);
    
        for ii = 1:length(data)
            b = mod(round(phi_list(ii)/360*bin),bin)+1;
            esum(b) = esum(b) + error(ii);
            enum(b) = enum(b) + 1;
        end
        emean = esum./enum;
        for ii = 1:length(data)
            b = mod(round(phi_list(ii)/360*bin),bin)+1;
            erms(b) = erms(b) + (error(ii) - emean(b))^2;
        end
        erms = sqrt(erms./enum);

        if(~isempty(erange))
            eid = (phi_list >= erange(1)) & (phi_list <= erange(2));
            edata = error(eid);
        end
        
        if(disp)
            subplot(2,1,1);
    
            yyaxis left;
            plot(phi_list,data,'k.');
            axis([0,360,min(data),max(data)]);
            ylabel('data');

            yyaxis right;
            plot(phi_list,error,'r.');
            hold on;
            plot(phase_code,emean,'b-');
            axis([0,360,min(error),max(error)]);
            ylabel('error');
    
            legend('data','error');
            xlabel('phase(deg)');

            if(~isempty(erange))
                plot(phi_list(eid),error(eid),'m.');
            end
    
    
            subplot(2,1,2);
            bar(phase_code,erms);
            axis([0,360,0,max(erms)*1.1]);
            xlabel('phase(deg)');
            ylabel('RMS error');
        end
    end

    if(isempty(erange))
        edata = [];
    end

end