function validateAoutData(aout_data)
%VALIDATEAOUTDATA Validate analog output data format
%
%   validateAoutData(aout_data)
%
% Throws error if data is invalid with descriptive message

if ~isnumeric(aout_data)
    error('Data must be numeric, got %s', class(aout_data));
end

if ~isreal(aout_data)
    error('Data must be real-valued, got complex numbers');
end

if any(isnan(aout_data(:)))
    error('Data contains NaN values');
end

if any(isinf(aout_data(:)))
    error('Data contains Inf values');
end

if isempty(aout_data)
    error('Data is empty');
end

if isvector(aout_data)
    nSamples = length(aout_data);
else
    nSamples = size(aout_data, 2);
end

if nSamples < 100
    error('Insufficient samples (%d), need at least 100', nSamples);
end

data_range = max(aout_data(:)) - min(aout_data(:));
if data_range == 0
    error('Data is constant (no variation)');
end

if data_range < 1e-10
    error('Data range too small (%.2e), likely invalid', data_range);
end

end
