function bin = findBin(Fs, Fin, N)
%FINDBIN Find coherent FFT bin for a given signal frequency (legacy)
%   This is a legacy wrapper for backward compatibility. New code should
%   use FINDBIN (lowercase) instead.
%
%   Legacy interface:
%     bin = FINDBIN(Fs, Fin, N)
%
%   See also: findbin

    % Call new function with lowercase name
    bin = findbin(Fs, Fin, N);

end
