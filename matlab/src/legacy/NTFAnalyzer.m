function [noiSup] = NTFAnalyzer(NTF,Flow,Fhigh,isPlot)
%NTFANALYZER Analyze noise transfer function performance (legacy)
%   DEPRECATED: This function is maintained for backward compatibility.
%   Please use ntfperf (lowercase) instead.
%
%   This function is a wrapper that calls ntfperf with the same
%   functionality. All new code should use ntfperf directly.
%
%   Legacy interface:
%     [noiSup] = NTFANALYZER(NTF, Flow, Fhigh, isPlot)
%
%   Inputs:
%     NTF - Noise transfer function in z-domain
%       Now named 'ntf' in the new function
%     Flow - Low bound frequency of signal band (relative to Fs)
%       Now named 'fl' in the new function
%     Fhigh - High bound frequency of signal band (relative to Fs)
%       Now named 'fh' in the new function
%     isPlot - Display plot switch (optional, default: 0)
%       Now named 'disp' in the new function
%
%   Outputs:
%     noiSup - SNR improvement from noise-shaping and over-sampling (dB)
%       Now named 'snr' in the new function
%
%   See also: ntfperf

    % Call the new ntfperf function with appropriate arguments
    if nargin < 4
        noiSup = ntfperf(NTF, Flow, Fhigh);
    else
        noiSup = ntfperf(NTF, Flow, Fhigh, isPlot);
    end

end