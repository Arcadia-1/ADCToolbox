function overflowChk(raw_code,weight,OFB)

if(nargin < 2)
    OFB = 1;
end

[N,M] = size(raw_code);

data_decom = zeros([N,M]);
range_min = zeros([1,M]);
range_max = zeros([1,M]);

for ii = 1:M
    tmp = raw_code(:,ii:end)*weight(ii:end)';
    
%     [~,im] = max(tmp);
%     tmp(im) = (2^(M-ii+1)-1)/2;
%     [~,im] = min(tmp);
%     tmp(im) = (2^(M-ii+1)-1)/2;
    
    data_decom(:,ii) = tmp / sum(weight(ii:end));
    range_min(ii) = min(tmp) / sum(weight(ii:end));
    range_max(ii) = max(tmp) / sum(weight(ii:end));
end

ovf_zero = (data_decom(:,M-OFB+1) <= 0);     
ovf_one = (data_decom(:,M-OFB+1) >= 1);      
non_ovf = ~(ovf_zero | ovf_one);


hold on;
plot([0,M+1],[1,1],'-k');
plot([0,M+1],[0,0],'-k');
plot((1:M),range_min,'-r');
plot((1:M),range_max,'-r');
for ii = 1:M

    h = scatter(ones([1,N])*ii, data_decom(:,ii), 'MarkerFaceColor','b','MarkerEdgeColor','b'); 
    h.MarkerFaceAlpha = 0.01;
    h.MarkerEdgeAlpha = 0.01;
    
    h = scatter(ones([1,sum(ovf_one)])*ii-0.2, data_decom(ovf_one,ii), 'MarkerFaceColor','r','MarkerEdgeColor','r'); 
    h.MarkerFaceAlpha = 0.01;
    h.MarkerEdgeAlpha = 0.01;
    
    h = scatter(ones([1,sum(ovf_zero)])*ii+0.2, data_decom(ovf_zero,ii), 'MarkerFaceColor','y','MarkerEdgeColor','y'); 
    h.MarkerFaceAlpha = 0.01;
    h.MarkerEdgeAlpha = 0.01;
    
    text(ii, -0.05, [num2str(sum(data_decom(:,ii) <= 0)/N*100,'%.1f'),'%']);
    text(ii, 1.05, [num2str(sum(data_decom(:,ii) >= 1)/N*100,'%.1f'),'%']);
end

axis([0,M+1,-0.1,1.1]);
xticks(1:M);
xticklabels(M:-1:0);
xlabel('bit');
ylabel('Residue Distribution');