function validateDoutData(bits)
%VALIDATEDOUTDATA Validate digital output (bits) data format
%
%   validateDoutData(bits)
%
% Throws error if data is invalid with descriptive message
% See user_preference.md for validation thresholds

% Validation thresholds (see user_preference.md)
MIN_SAMPLES = 100;
MIN_BITS = 2;
MAX_BITS = 32;

% Check basic types
if ~isnumeric(bits) || ~isreal(bits)
    error('[Data must be real numeric, got %s]', class(bits));
end

if any(isnan(bits(:))) || any(isinf(bits(:)))
    error('[Data contains NaN or Inf values]');
end

if isempty(bits)
    error('[Data is empty]');
end

% Check matrix format
if ~ismatrix(bits) || isvector(bits)
    error('[Data must be 2D matrix (N×B), got size %s]', num2str(size(bits)));
end

% Check binary values
if ~all(ismember(bits(:), [0, 1]))
    unique_vals = unique(bits(:));
    error('[Data must be binary (0 or 1), found: %s]', mat2str(unique_vals'));
end

% Check dimensions
[nSamples, nBits] = size(bits);
if nSamples < MIN_SAMPLES
    error('[Insufficient samples: %d, need ≥%d]', nSamples, MIN_SAMPLES);
end

if nBits < MIN_BITS
    error('[Insufficient bits: %d, need ≥%d]', nBits, MIN_BITS);
end

if nBits > MAX_BITS
    warning('[Unusual bit count: %d, expected <%d]', nBits, MAX_BITS);
end

% Check for stuck bits
stuck_low = find(sum(bits, 1) == 0);
stuck_high = find(sum(bits, 1) == nSamples);

if ~isempty(stuck_low)
    warning('[Bit(s) stuck at 0: %s]', mat2str(stuck_low));
end

if ~isempty(stuck_high)
    warning('[Bit(s) stuck at 1: %s]', mat2str(stuck_high));
end

end
