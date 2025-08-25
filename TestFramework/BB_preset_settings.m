pr.chip_id = 16;
pr.test_type = 'NoTest';

% parameters below are related to your actual testing plan

pr.F_clk = 500 * 10^6;
pr.P_clk = -10;  % dBm

pr.T_read = 2*10^-6;

pr.F_io = 20 * 10^6;
pr.T_io_shift = -0.2 * 10^-6;

pr.N_ex_sam = 100;

pr.weight_MSB = 31.76;

pr.V_in_comp = 0;

pr.AVDD_scalling = 1;

pr.single_end = 0;


