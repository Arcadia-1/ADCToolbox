function overflowChk(raw_code,weight,OFB)
%OVERFLOWCHK Check ADC overflow by analyzing bit segment residue distributions (legacy)
%   DEPRECATED: This function is maintained for backward compatibility.
%   Please use bitchk instead.
%
%   This function is a wrapper that calls bitchk with the same
%   functionality. All new code should use bitchk directly.
%
%   Legacy interface:
%     OVERFLOWCHK(raw_code, weight)
%     OVERFLOWCHK(raw_code, weight, OFB)
%
%   Inputs:
%     raw_code - Raw ADC output codes matrix
%       Now named 'bits' in the new function
%     weight - Bit weights for ADC code calculation
%       Now named 'wgt' in the new function
%     OFB - Bit position to check for overflow (optional)
%       Now named 'chkpos' in the new function
%
%   Outputs:
%     None (displays a visualization plot)
%
%   See also: bitchk

    % Call the new bitchk function with appropriate arguments
    if nargin < 3
        bitchk(raw_code, weight);
    else
        bitchk(raw_code, weight, OFB);
    end

end
