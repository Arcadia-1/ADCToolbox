%% Test plotwgt radix output shape
% plotwgt returns radix aligned to bit indices: radix(1) is NaN and
% radix(i) is the adjacent ratio between weights i-1 and i.

weights = [8 4 2 1];
[radix, wgtsca, effres] = plotwgt(weights, 0);

assert(isequal(size(radix), size(weights)), ...
    'plotwgt radix must have one element per input bit');
assert(isnan(radix(1)), ...
    'plotwgt radix(1) must be NaN because the MSB has no previous bit');
assert(max(abs(radix(2:end) - [2 2 2])) < 1e-12, ...
    'plotwgt radix(2:end) must contain adjacent weight ratios');
assert(abs(wgtsca - 1) < 1e-12, ...
    'plotwgt scaling for binary integer weights should remain 1');
assert(abs(effres - 4) < 1e-12, ...
    'plotwgt effective resolution for [8 4 2 1] should be 4 bits');
