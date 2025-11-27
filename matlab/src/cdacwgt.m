function [weight, ctot] = cdacwgt(cd, cb, cp)
%CDACWGT Calculate bit weights for multi-segment capacitive DAC
%   This function computes the normalized bit weights of a multi-segment
%   capacitive digital-to-analog converter (CDAC) with bridge capacitors
%   and parasitic capacitances. The calculation accounts for capacitive
%   divider effects and weight attenuation through each segment.
%
%   The function processes bits from LSB to MSB, calculating how each bit's
%   voltage contribution is attenuated by the capacitive network. Bridge
%   capacitors (cb) separate segments, and parasitic capacitances (cp)
%   appear in parallel with each DAC capacitor.
%
%   Syntax:
%     [weight, ctot] = CDACWGT(cd, cb, cp)
%
%   Inputs:
%     cd - DAC bit capacitors, ordered [MSB ... LSB]
%       Vector of positive real numbers
%       Size: 1 x M (where M is number of bits)
%     cb - Bridge capacitors between segments, ordered [MSB ... LSB]
%       Vector of non-negative real numbers
%       Size: 1 x M (same length as cd)
%       Use 0 for bits without bridge capacitor (within same segment)
%     cp - Parasitic capacitors for each bit, ordered [MSB ... LSB]
%       Vector of non-negative real numbers
%       Size: 1 x M (same length as cd)
%
%   Outputs:
%     weight - Normalized bit weights representing voltage gain from
%       each Vbot to DAC's output, ordered [MSB ... LSB]
%       Vector of positive real numbers
%       Size: 1 x M (same size as inputs)
%       Range: [0, 1]
%     ctot - The total capacitance seen from MSB side
%       Scalar positive real number
%
%   Circuit Model (for one bit):
%     MSB side <---||------------||---< LSB side
%                  cb   |    |   Cl (load from previous bits)
%                      ---  ---
%                  cp  ---  ---  cd
%                       |    |
%                      gnd   Vbot (input voltage on the bottom plate)
%
%   Examples:
%     % Simple binary-weighted 4-bit DAC with no bridges or parasitics
%     cd = [8 4 2 1];           % Binary-weighted capacitors [MSB ... LSB]
%     cb = [0 0 0 0];           % No bridge caps
%     cp = [0 0 0 0];           % No parasitics
%     [weight, ctot] = cdacwgt(cd, cb, cp)
%     % Returns: weight = [0.5333 0.2667 0.1333 0.0667], ctot = 15
%
%     % 6-bit CDAC with 3+3 binary-weighted segments
%     cd = [4 2 1   4 2 1];     % Two 3-bit segments [MSB ... LSB]
%     cb = [0 4 0 8/7 0 0];   % Bridge cap between segments
%     cp = [0 0 0   0 0 1];     % cp=1 at the LSB to refine the weight
%     [weight, ctot] = cdacwgt(cd, cb, cp)
%     % Returns: weight = [0.5000 0.2500 0.1250 0.0625 0.0312 0.0156]
%     %          ctot = 8
%
%   Notes:
%     - Input bits must be ordered from MSB to LSB
%     - Bridge capacitor cb(i) = 0 means bit i is in same segment as bit i+1
%     - Bridge capacitor cb(i) > 0 creates capacitive divider between segments
%     - Output capacitance ctot is useful for analyzing noise and loading effects
%
%   Algorithm:
%     For each bit from LSB to MSB (processing reversed input arrays):
%       1. Compute total capacitance of current bit: Ct = cp + cd + Cl (load from prev bits)
%       2. Attenuate all previous weights by factor Cl/Ct
%       3. Current bit weight = cd/Ct
%       4. Calculate load capacitance for next bit: Cl = Ct (if no bridge) or Cl = series(cb, Ct)
%
%   See also: cap2weight (legacy)

    % Input validation
    if nargin < 3
        error('cdacwgt:notEnoughInputs', ...
              'Three input arguments required: cd, cb, cp.');
    end

    if ~isvector(cd) || ~isvector(cb) || ~isvector(cp)
        error('cdacwgt:invalidInput', ...
              'All inputs must be vectors.');
    end

    M = length(cd);

    if length(cb) ~= M || length(cp) ~= M
        error('cdacwgt:sizeMismatch', ...
              'All input vectors must have the same length.');
    end

    if any(cd <= 0)
        error('cdacwgt:invalidCd', ...
              'DAC capacitors cd must be positive.');
    end

    if any(cb < 0) || any(cp < 0)
        error('cdacwgt:invalidCapacitance', ...
              'Bridge and parasitic capacitors must be non-negative.');
    end

    if ~isreal(cd) || ~isreal(cb) || ~isreal(cp)
        error('cdacwgt:invalidInput', ...
              'Capacitance values must be real numbers.');
    end

    % Reverse inputs to process LSB to MSB (internal algorithm requirement)
    cd = cd(end:-1:1);
    cb = cb(end:-1:1);
    cp = cp(end:-1:1);

    % Initialize weight vector
    weight = zeros(1, M);

    % Start with no load capacitance
    Cl = 0;

    % Process each bit from LSB to MSB
    for i = 1:M
        % Total capacitance for current bit
        Ct = cp(i) + cd(i) + Cl;

        % Attenuate previous weights due to capacitive division
        weight = weight * Cl / Ct;

        % Current bit weight from capacitive divider
        weight(i) = cd(i) / Ct;

        % Update load capacitance for next bit
        if cb(i) == 0
            % No bridge: load is the current total capacitance
            Cl = Ct;
        else
            % Bridge present: series combination of bridge and current total capacitance
            Cl = 1 / (1/cb(i) + 1/Ct);
        end
    end

    % Output capacitance is final load
    ctot = Cl;

    % Reverse weight output to match MSB to LSB ordering
    weight = weight(end:-1:1);

end