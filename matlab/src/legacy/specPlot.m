function [ENoB,SNDR,SFDR,SNR,THD,pwr,NF,h] = specPlot(data,varargin)
%SPECPLOT Plot power spectrum and calculate ADC performance metrics (legacy)
%   DEPRECATED: This function is maintained for backward compatibility.
%   Please use specplot (lowercase) instead.
%
%   This function is a wrapper that calls specplot with the same
%   functionality. All new code should use specplot directly.
%
%   Legacy interface:
%     [ENoB,SNDR,SFDR,SNR,THD,pwr,NF,h] = SPECPLOT(data)
%     [ENoB,SNDR,SFDR,SNR,THD,pwr,NF,h] = SPECPLOT(data, Fs)
%     [ENoB,SNDR,SFDR,SNR,THD,pwr,NF,h] = SPECPLOT(data, Fs, maxCode)
%     [ENoB,SNDR,SFDR,SNR,THD,pwr,NF,h] = SPECPLOT(data, Fs, maxCode, harmonic)
%     [ENoB,SNDR,SFDR,SNR,THD,pwr,NF,h] = SPECPLOT(data, 'Name', Value)
%
%   Inputs:
%     data - ADC output data samples
%       Same meaning in the new function (now called 'sig')
%
%   Optional Positional Inputs:
%     Fs - Sampling frequency in Hz
%       Same meaning in the new function
%     maxCode - Full scale range (max-min)
%       Same meaning in the new function (aliased to 'maxSignal')
%     harmonic - Number of harmonics to analyze
%       Same meaning in the new function
%
%   Name-Value Parameters:
%     'OSR' - Oversampling ratio
%       Same meaning in the new function
%     'winType' - Window function handle
%       Same meaning in the new function (aliased to 'window')
%     'sideBin' - Number of bins around signal peak
%       Same meaning in the new function
%     'label' - Enable plot annotations
%       Same meaning in the new function
%     'assumedSignal' - Override signal power in dB
%       Same meaning in the new function
%     'isPlot' - Enable plotting
%       Same meaning in the new function (aliased to 'disp')
%     'noFlicker' - Cutoff frequency for flicker noise removal
%       Same meaning in the new function
%     'nTHD' - Number of harmonics for THD calculation
%       Same meaning in the new function
%     'coAvg' - Enable coherent averaging
%       Same meaning in the new function
%     'NFMethod' - Noise floor estimation method
%       Same meaning in the new function
%
%   Outputs:
%     ENoB - Effective Number of Bits
%       Same meaning in the new function (now lowercase 'enob')
%     SNDR - Signal-to-Noise and Distortion Ratio in dB
%       Same meaning in the new function (now lowercase 'sndr')
%     SFDR - Spurious-Free Dynamic Range in dB
%       Same meaning in the new function (now lowercase 'sfdr')
%     SNR - Signal-to-Noise Ratio in dB
%       Same meaning in the new function (now lowercase 'snr')
%     THD - Total Harmonic Distortion in dB
%       Same meaning in the new function (now lowercase 'thd')
%     pwr - Signal power in dBFS
%       Same meaning in the new function (now 'sigpwr')
%     NF - Noise Floor in dB
%       Same meaning in the new function (now 'noi')
%     h - Plot handle or empty array if isPlot=0
%       Same meaning in the new function
%
%   See also: specplot

    % Call the new specplot function with all arguments
    % New function returns: [enob,sndr,sfdr,snr,thd,sigpwr,noi,nsd,h]
    % Legacy function returns: [ENoB,SNDR,SFDR,SNR,THD,pwr,NF,h]
    [ENoB,SNDR,SFDR,SNR,THD,pwr,NF,~,h] = specplot(data,varargin{:});

end
