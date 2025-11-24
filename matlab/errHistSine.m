function [emean, erms, phase_code, anoi, pnoi, err, xx] = errHistSine(data, varargin)

    % mode = 0 : phase as x axis; 
    % mode >= 1 : code as axis;

    % emean : mean error over phase / code
    % erms : RMS error over phase / code
    % phase_code : the list of phase / code corresponding to the two output lists above
    % anoi : amplitude noise (reference noise)
    % pnoi : phase noise (phase jitter)
    % err : raw errors of each data point
    % xx : x-axies values (phase / code) corresponding to the raw err

    p = inputParser;
    addOptional(p, 'bin', 100, @(x) isnumeric(x) && isscalar(x) && (x > 0));
    addOptional(p, 'fin', 0, @(x) isnumeric(x) && isscalar(x) && (x > 0) && (x < 1));
    addOptional(p, 'disp', 1);
    addOptional(p, 'mode', 0, @(x) isnumeric(x));
    addParameter(p, 'erange', []);  % err filter, only the erros in erange are return to err
    parse(p, varargin{:});
    bin = round(p.Results.bin);
    fin = p.Results.fin;
    disp = p.Results.disp;
    codeMode = p.Results.mode;
    erange = p.Results.erange;

    [N,M] = size(data);
    if(M == 1)
        data = data'; 
    end

    if(fin == 0)
        [data_fit,fin,mag,~,phi] = sineFit(data);
    else
        [data_fit,~,mag,~,phi] = sineFit(data,fin);
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

        anoi = nan;
        pnoi = nan;

        if(~isempty(erange))
            eid = (xx >= erange(1)) & (xx <= erange(2));
            xx = xx(eid);
            err = err(eid);
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
                plot(xx,err,'m.');
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


        asen = abs(cos(phase_code/360*2*pi)).^2;    % amplitude noise sensitivity
        psen = abs(sin(phase_code/360*2*pi)).^2;    % phase noise sensitivity

        tmp = linsolve([asen',psen',ones(bin,1)], erms'.^2);

        anoi = sqrt(tmp(1));
        pnoi = sqrt(tmp(2))/mag;
        ermsbl = tmp(3);    % erms baseline

        if(anoi < 0 || imag(anoi) ~= 0)     % pnoise only
            tmp = linsolve([psen',ones(bin,1)], erms'.^2);
            anoi = 0;
            pnoi = sqrt(tmp(1))/mag;
            ermsbl = tmp(2);  

            if(pnoi < 0 || imag(pnoi) ~= 0)
                anoi = 0;
                pnoi = 0;
                ermsbl = mean(erms.^2);
            end
        end

        if(pnoi < 0 || imag(pnoi) ~= 0)     % anoise only
            tmp = linsolve([asen',ones(bin,1)], erms'.^2);
            pnoi = 0;
            anoi = sqrt(tmp(1));
            ermsbl = tmp(2);      

            if(anoi < 0 || imag(anoi) ~= 0)
                anoi = 0;
                pnoi = 0;
                ermsbl = mean(erms.^2);
            end
        end

        if(~isempty(erange))
            eid = (xx >= erange(1)) & (xx <= erange(2));
            xx = xx(eid);
            err = err(eid);
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
                plot(xx,err,'m.');
            end
    
            subplot(2,1,2);
            bar(phase_code,erms);
            hold on;
            plot(phase_code, sqrt((anoi.^2)*asen + ermsbl), 'b-', 'LineWidth',2);
            plot(phase_code, sqrt((pnoi.^2)*psen*(mag^2) + ermsbl), 'r-', 'LineWidth',2);
            axis([0,360,0,max(erms)*1.2]);
            text(10, max(erms)*1.15, sprintf('Normalized Amplitude Noise RMS = %.2d',anoi/mag), 'color', [0,0,1]);
            text(10, max(erms)*1.05, sprintf('Phase Noise RMS = %.2d rad',pnoi), 'color', [1,0,0]);
            xlabel('phase(deg)');
            ylabel('RMS error');
        end
    end

end