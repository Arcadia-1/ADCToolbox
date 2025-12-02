function [h] = specPlotPhase(data,varargin)
%SPECPLOTPHASE Plot coherent phase spectrum with polar display (legacy)
%   DEPRECATED: This function is maintained for backward compatibility.
%   Please use phaseplot (lowercase) instead.
%
%   This function is a wrapper that calls phaseplot with the same
%   functionality. All new code should use phaseplot directly.
%
%   Legacy interface:
%     h = SPECPLOTPHASE(data)
%     h = SPECPLOTPHASE(data, N_fft)
%     h = SPECPLOTPHASE(data, N_fft, harmonic)
%     h = SPECPLOTPHASE(data, 'Name', Value)
%
%   Inputs:
%     data - Signal to be analyzed
%       In the new function, this parameter is named 'sig'
%
%   Optional Positional Inputs:
%     N_fft - FFT length (DEPRECATED - no longer used)
%       The new function always uses length(data) as FFT length
%       This parameter is now interpreted as Fs (sampling frequency)
%       for backward compatibility approximation
%     harmonic - Number of harmonics to display
%       Same meaning in the new function
%
%   Name-Value Parameters:
%     'OSR' - Oversampling ratio
%       Same meaning in the new function
%     'cutoff' - High-pass cutoff frequency for low-frequency noise removal in Hz
%       New parameter available in phaseplot
%
%   Outputs:
%     h - Plot handle
%       Same meaning in the new function
%
%   Migration Notes:
%     - Parameter 'data' is now named 'sig' in phaseplot
%     - The N_fft parameter has been removed in the new phaseplot function
%     - New phaseplot adds Fs (sampling frequency) as second positional parameter
%     - FFT length is now always determined by the data length
%     - New 'cutoff' parameter available to remove low-frequency noise
%     - If you were using N_fft, ensure your data has the desired length
%     - Old code: h = specPlotPhase(data, 1024, 5)
%     - New code: h = phaseplot(sig, Fs, 5) where Fs is sampling frequency
%
%   See also: phaseplot

    % Call the new phaseplot function with all arguments
    % Note: The second positional parameter is now interpreted as harmonic instead of N_fft
    % Force FFT mode for backward compatibility (new phaseplot defaults to LMS mode)
    h = phaseplot(data, varargin{:}, 'mode', 'FFT');

end
