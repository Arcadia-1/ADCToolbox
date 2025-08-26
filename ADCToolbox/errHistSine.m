function [emean, erms, phase_code, edata, err, xx] = errHistSine(data, varargin)

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

    err = data_fit-data;
    
    if(codeMode)
        xx = data;
        dat_min = min(data);
        dat_max = max(data);
        bin_wid = (dat_max-dat_min)/bin;
        phase_code =  min(data) + [1:bin]*bin_wid - bin_wid/2;
        
        enum = zeros([1,bin]);
        esum = zeros([1,bin]);
        erms = zeros([1,bin]);
    
        for ii = 1:length(data)
            b = min(floor((data(ii)-dat_min)/bin_wid)+1,bin);
            esum(b) = esum(b) + err(ii);
            enum(b) = enum(b) + 1;
        end
        emean = esum./enum;
        for ii = 1:length(data)
            b = min(floor((data(ii)-dat_min)/bin_wid)+1,bin);
            erms(b) = erms(b) + (err(ii) - emean(b))^2;
        end
        erms = sqrt(erms./enum);

        if(~isempty(erange))
            eid = (data >= erange(1)) & (data <= erange(2));
            edata = err(eid);
        end

        if(disp)
            subplot(2,1,1);
    
            plot(data,err,'r.');
            hold on;
            plot(phase_code,emean,'b-');
            axis([dat_min,dat_max,min(err),max(err)]);
            ylabel('error');
            xlabel('code');

            if(~isempty(erange))
                plot(data(eid),err(eid),'m.');
            end
    
            subplot(2,1,2);
            bar(phase_code,erms);
            axis([dat_min,dat_max,0,max(erms)*1.1]);
            xlabel('code');
            ylabel('RMS error');
        end

    else
        xx = mod(phi/pi*180 + (0:length(data)-1)*fin*360,360);
        phase_code = (0:bin-1)/bin*360;

        enum = zeros([1,bin]);
        esum = zeros([1,bin]);
        erms = zeros([1,bin]);
    
        for ii = 1:length(data)
            b = mod(round(xx(ii)/360*bin),bin)+1;
            esum(b) = esum(b) + err(ii);
            enum(b) = enum(b) + 1;
        end
        emean = esum./enum;
        for ii = 1:length(data)
            b = mod(round(xx(ii)/360*bin),bin)+1;
            erms(b) = erms(b) + (err(ii) - emean(b))^2;
        end
        erms = sqrt(erms./enum);

        if(~isempty(erange))
            eid = (xx >= erange(1)) & (xx <= erange(2));
            edata = err(eid);
        end
        
        if(disp)
            subplot(2,1,1);
    
            yyaxis left;
            plot(xx,data,'k.');
            axis([0,360,min(data),max(data)]);
            ylabel('data');

            yyaxis right;
            plot(xx,err,'r.');
            hold on;
            plot(phase_code,emean,'b-');
            axis([0,360,min(err),max(err)]);
            ylabel('error');
    
            legend('data','error');
            xlabel('phase(deg)');

            if(~isempty(erange))
                plot(xx(eid),err(eid),'m.');
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