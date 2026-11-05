%% Regression checks for inlsin code labels and code-width normalization

fprintf('[%s] missing-code labels and code-width normalization checks\n', mfilename);

% A coherent sine covers N uniform phases and overdrives codes 0 and 255.
N = 2^20;
n = 0:N-1;
x = 128 + 127.9 * sin(2*pi*1237*n/N);
tol = 2e-3;  % LSB; this record recovers each code width to < 1e-3 LSB

%% A missing code is reported at its own code.
widths = ones(1, 254);  % widths(c) is the width of internal code c
widths(64) = 0;
widths(65) = 2;         % code 65 absorbs code 64
[inl, dnl, code] = inlsin(inlsin_width_codes(x, widths), 0, false);

assert(isequal(code, 1:254), 'inlsin must report the internal codes 1..254.');
assert(isequal(code(dnl == -1), 64), 'Missing code 64 must be reported at code 64.');
idx = find(ismember(code, 63:66));
assert(max(abs(dnl(idx) - [0, -1, 1, 0])) < tol, 'DNL is misaligned around missing code 64.');
assert(max(abs(inl(idx) - [0, -1, 0, 0])) < tol, 'INL is misaligned around missing code 64.');
assert_inlsin_widths(widths, code, dnl, inl, tol, 'one missing code');

%% Many missing codes keep their labels and the INL ends at zero.
missing = 16:4:236;
widths = ones(1, 254);
widths(missing) = 0;
widths(missing + 1) = 2;
[inl, dnl, code] = inlsin(inlsin_width_codes(x, widths), 0, false);

assert(isequal(code(dnl == -1), missing), 'Missing codes must be reported at their own codes.');
assert_inlsin_widths(widths, code, dnl, inl, tol, 'many missing codes');

%% Nonuniform widths without missing codes are not shifted.
widths = repmat([0.6, 1.4], 1, 127);
[inl, dnl, code] = inlsin(inlsin_width_codes(x, widths), 0, false);

assert(isequal(code, 1:254), 'inlsin must report the internal codes 1..254.');
assert_inlsin_widths(widths, code, dnl, inl, tol, 'alternating widths');

%% With exclusion, one LSB is the mean width of the reported codes.
widths = 1 + 0.3 * cos(2*pi*(0:253)/253);  % wider excluded edge codes
missing = [40, 41, 120, 200];
widths(missing) = 0;
widths = widths * 254 / sum(widths);
data = inlsin_width_codes(x, widths);

% excl=0.01 and 0.05 exclude round(2.55) = 3 and round(12.75) = 13 codes at
% each end; the next codes inward absorb the clipped samples and are not
% reported either.
exclusions = [0.01, 4, 251; 0.05, 14, 241];
for ii = 1:size(exclusions, 1)
    excl = exclusions(ii, 1);
    [inl, dnl, code] = inlsin(data, excl, false);

    assert(isequal(code, exclusions(ii, 2):exclusions(ii, 3)), ...
        'excl=%.2f must report codes %d..%d.', excl, exclusions(ii, 2), exclusions(ii, 3));
    assert(isequal(code(dnl == -1), missing), ...
        'excl=%.2f: missing codes must be reported at their own codes.', excl);
    assert_inlsin_widths(widths, code, dnl, inl, tol, sprintf('excl=%.2f', excl));
end

function data = inlsin_width_codes(x, widths)
% Quantize x with an ADC whose internal code c is widths(c) LSB wide.
    lower_edges = 1 + [0, cumsum(widths)];
    data = zeros(size(x));
    for k = 1:numel(lower_edges)
        data = data + (x >= lower_edges(k));
    end
end

function assert_inlsin_widths(widths, code, dnl, inl, tol, label)
% Compare with the true widths, using their mean over code as one LSB.
    w = widths(code);
    expected_dnl = w / mean(w) - 1;
    assert(max(abs(dnl - expected_dnl)) < tol, '%s: DNL does not match the code widths.', label);
    assert(max(abs(inl - cumsum(expected_dnl))) < tol, '%s: INL does not match the code widths.', label);
    assert(abs(mean(dnl)) < 1e-12, '%s: DNL must have zero mean.', label);
    assert(abs(inl(end)) < 1e-9, '%s: INL must end at zero.', label);
end
