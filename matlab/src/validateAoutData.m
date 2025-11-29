function validateAoutData(aout_data)
%VALIDATEAOUTDATA Validate analog output data format
%
%   validateAoutData(aout_data)
%
% Throws error if data is invalid with descriptive message
% See user_preference.md for validation thresholds

% Validation thresholds (see user_preference.md)
MIN_SAMPLES = 100;
MIN_RANGE = 1e-10;

% Check basic types
if ~isnumeric(aout_data) || ~isreal(aout_data)
    error('[Data must be real numeric, got %s]', class(aout_data));
end

if any(isnan(aout_data(:))) || any(isinf(aout_data(:)))
    error('[Data contains NaN or Inf values]');
end

if isempty(aout_data)
    error('[Data is empty]');
end

% Check sample count
nSamples = numel(aout_data);
if nSamples < MIN_SAMPLES
    error('[Insufficient samples: %d, need ≥%d]', nSamples, MIN_SAMPLES);
end

% Check signal variation
data_range = max(aout_data(:)) - min(aout_data(:));
if data_range < MIN_RANGE
    error('[No signal variation: range=%.2e, need >%.2e]', data_range, MIN_RANGE);
end

end
