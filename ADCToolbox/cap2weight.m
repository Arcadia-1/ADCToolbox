function [weight,Co] = cap2weight(Cd, Cb, Cp)

%   Cd = DAC bit cap,  [LSB ... MSB]
%   Cb = bridge cap, [LSB .. MSB]
%   Cp = parastic cap, [LSB ... MSB]
%   weight = gain from Vi to Vout, [LSB ... MSB]
%   Co = output cap
%     
%     MSB <---||--------||---< LSB
%           Cb   |    |   Cl
%               ---  --- 
%           Cp  ---  ---  Cd
%                |    |
%               gnd   Vi   

    M = length(Cd);
    weight = zeros(1,M);

    Cl = 0;
    for i1 = 1:M
        Cs = Cp(i1) + Cd(i1) + Cl;
        weight = weight * Cl/Cs;
        weight(i1) = Cd(i1)/Cs;
        if(Cb(i1) == 0)
            Cl = Cs;
        else
            Cl = 1/(1/Cb(i1) + 1/Cs);
        end
    end
    
    Co = Cl;
    
end