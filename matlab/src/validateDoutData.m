function validateDoutData(bits)
%VALIDATEDOUTDATA Validate digital output (bits) data format
%
%   validateDoutData(bits)
%
% Throws error if data is invalid with descriptive message

if ~isnumeric(bits)
    error('Data must be numeric, got %s', class(bits));
end

if ~isreal(bits)
    error('Data must be real-valued, got complex numbers');
end

if any(isnan(bits(:)))
    error('Data contains NaN values');
end

if any(isinf(bits(:)))
    error('Data contains Inf values');
end

if isempty(bits)
    error('Data is empty');
end

if ~ismatrix(bits) || isvector(bits)
    error('Data must be 2D matrix (N samples x B bits), got size [%s]', num2str(size(bits)));
end

unique_vals = unique(bits(:));
if ~all(ismember(unique_vals, [0, 1]))
    error('Data must contain only binary values (0 or 1), found values: %s', mat2str(unique_vals'));
end

[nSamples, nBits] = size(bits);

if nSamples < 100
    error('Insufficient samples (%d), need at least 100', nSamples);
end

if nBits < 2
    error('Insufficient bits (%d), need at least 2', nBits);
end

if nBits > 32
    warning('Unusual bit count (%d), verify this is correct', nBits);
end

stuck_bits = sum(bits, 1);
all_zero_bits = find(stuck_bits == 0);
all_one_bits = find(stuck_bits == nSamples);

if ~isempty(all_zero_bits)
    warning('Bit(s) stuck at 0: %s', mat2str(all_zero_bits));
end

if ~isempty(all_one_bits)
    warning('Bit(s) stuck at 1: %s', mat2str(all_one_bits));
end

end
