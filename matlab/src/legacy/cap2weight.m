function [weight, Co] = cap2weight(Cd, Cb, Cp)
%CAP2WEIGHT Calculate bit weights for multi-segment capacitive DAC (legacy)
%   This is a legacy wrapper for backward compatibility. New code should
%   use CDACWGT instead.
%
%   Legacy interface with LSB-to-MSB ordering:
%     [weight, Co] = CAP2WEIGHT(Cd, Cb, Cp)
%
%   Inputs (ordered [LSB ... MSB]):
%     Cd - DAC bit capacitors
%     Cb - Bridge capacitors between segments
%     Cp - Parasitic capacitors for each bit
%
%   Outputs:
%     weight - Normalized bit weights (ordered [LSB ... MSB])
%     Co - Total output capacitance
%
%   See also: cdacwgt

    % Convert from legacy LSB-to-MSB ordering to new MSB-to-LSB ordering
    cd = Cd(end:-1:1);
    cb = Cb(end:-1:1);
    cp = Cp(end:-1:1);

    % Call new function
    [weight_new, Co] = cdacwgt(cd, cb, cp);

    % Convert weight output back to legacy LSB-to-MSB ordering
    weight = weight_new(end:-1:1);

end