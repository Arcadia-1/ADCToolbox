pr.vtol = 0.01; % voltage tolerance (don't need to change in general)

% parameters below are related to your actual testing plan

pr.Vana = 1.4;
pr.Vamp = 0.95;
pr.Vref1 = 1.2;
pr.Vref2 = 0.35;
pr.Vdig1 = 1.22;
pr.Vdig2 = 0.9;
pr.Vdig3 = 0.975;
pr.Viol = 1.0;
pr.Vioh = 1.2;
pr.Vclk = 1.0;
pr.Vcm = 0.45;

pr.Iref1 = -5; % uA
pr.Iref2 = -0;

update_reference; 