function radix = weightScaling(weights)
%WEIGHTSCALING Visualize absolute bit weights with radix annotations (legacy)
%   DEPRECATED: This function is maintained for backward compatibility.
%   Please use plotwgt instead.
%
%   This function is a wrapper that calls plotwgt with the same
%   functionality. All new code should use plotwgt directly.
%
%   Legacy interface:
%     radix = WEIGHTSCALING(weights)
%
%   Inputs:
%     weights - Bit weights from MSB to LSB
%       Vector (1 x B), where B is the number of bits
%
%   Outputs:
%     radix - Radix between consecutive bits
%       Vector (1 x B-1), where B is the number of bits
%       radix(i) = |weight(i) / weight(i+1)|
%
%   See also: plotwgt, wcalsin

    % Call the new plotwgt function with all arguments
    radix = plotwgt(weights);

end
