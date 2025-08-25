function vpp_new = findVinpp(vpp_old, signal, signal_target)

if nargin < 3
    signal_target = -0.5;
end

vpp_new = vpp_old /  10^(signal/20) * 10^(signal_target/20);
