function overflowChk(raw_code,weight,OFB)

[N,M] = size(raw_code);
if(N < M)
    raw_code = raw_code';
    [N,M] = size(raw_code);
end

if(nargin < 3)
    OFB = M;
end

data_decom = zeros([N,M]);
range_min = zeros([1,M]);
range_max = zeros([1,M]);

for ii = 1:M
    tmp = raw_code(:,ii:end)*weight(ii:end)';
    
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

    h = scatter(ones([1,sum(non_ovf)])*ii, data_decom(non_ovf,ii), 'MarkerFaceColor','b','MarkerEdgeColor','b'); 
    h.MarkerFaceAlpha = min(max(10/N,0.01),1);
    h.MarkerEdgeAlpha = min(max(10/N,0.01),1);
    
    h = scatter(ones([1,sum(ovf_one)])*ii-0.2, data_decom(ovf_one,ii), 'MarkerFaceColor','r','MarkerEdgeColor','r'); 
    h.MarkerFaceAlpha = min(max(10/N,0.01),1);
    h.MarkerEdgeAlpha = min(max(10/N,0.01),1);
    
    h = scatter(ones([1,sum(ovf_zero)])*ii+0.2, data_decom(ovf_zero,ii), 'MarkerFaceColor','y','MarkerEdgeColor','y'); 
    h.MarkerFaceAlpha = min(max(10/N,0.01),1);
    h.MarkerEdgeAlpha = min(max(10/N,0.01),1);
    
    text(ii, -0.05, [num2str(sum(data_decom(:,ii) <= 0)/N*100,'%.1f'),'%']);
    text(ii, 1.05, [num2str(sum(data_decom(:,ii) >= 1)/N*100,'%.1f'),'%']);
end

axis([0,M+1,-0.1,1.1]);
xticks(1:M);
xticklabels(M:-1:0);
xlabel('bit');
ylabel('Residue Distribution');