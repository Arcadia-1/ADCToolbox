function errEnvelopeSpectrum(err_data, varargin)

p = inputParser;
addParameter(p, "Fs", 1);
parse(p, varargin{:});
Fs = p.Results.Fs;

e = err_data(:);
env = abs(hilbert(e)); % envelope = |hilbert|

specPlot(env, "Fs", Fs, "label", 0); % use toolbox spectrum plot

grid on;
xlabel("Frequency (Hz)");
ylabel("Envelope Spectrum (dB)");
% title("Error Envelope Spectrum")
end
