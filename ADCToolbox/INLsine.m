function [INL, DNL, code] = INLsine(data, clip)

    if(nargin < 3)
        clip = 0.01;    % INL DNL 误差排除比例
    end

    S = size(data);
    if(S(1) < S(2))
        data = data';
    end
    
    max_data = ceil(max(data));
    min_data = floor(min(data));
    max_data = round(max_data - clip*(max_data-min_data)/2);
    min_data = round(min_data + clip*(max_data-min_data)/2);
  
    code = min_data:max_data;
    data = min(max(data,min_data),max_data);

    DCC = hist(data,code);
    DCC = -cos(pi*cumsum(DCC)/sum(DCC));
    DNL = DCC(2:end) - DCC(1:end-1);
    code = code(1:end-1);
    
    clip = floor(clip * (max_data-min_data+1) / 2);
    code = code(clip+1:end-clip);
    DNL = DNL(clip+1:end-clip);
    
    DNL = DNL./sum(DNL);
    DNL = DNL*(max_data-min_data-clip*2+1)-1;
    DNL = DNL-mean(DNL);
    
    INL = cumsum(DNL);  

end