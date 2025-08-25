function RET = top_SAR_ADC_example(PAR)

% Simple SAR ADC example
% By Lu Jie

tic();
% ---------------------- simulation switches ------------------------------

NOISE = 1;          % enable noise
MISMATCH = 1;       % enable mismatch
OFFSET = 1;
DITHER = 0;

% ---------------------- architecture parameters --------------------------

n_SAR = 12;    % SAR ADC resolution

d_wgt_nom = 2.^[11:-1:0];   % nominal digital weight
n_DAC_Cd =  2.^[11:-1:0];   % CDAC cap number of each bit
n_DAC_Cb =  zeros([1,12]);  % CDAC bridge cap on each bit (0 for no bridge)
n_DAC_Cp =  zeros([1,12]);  % CDAC parasitic cap on each bit
s
v_ref = 5;          % reference voltage
v_cm = 2.5;         % common mode voltage

% ---------------------- physical parameters ------------------------------

phy_temp = 27;          % temperature
phy_cap_DAC = 4*10^-15; % unit DAC capacitance
sgm_mis_cap = 0.05;         % DAC mismatch of unit cap
sgm_os_cmp = 5*10^-3;       % comparator offset 
sgm_noi_cmp = NOISE * 0.25*10^-3;

% ------------------------ simulation settings ----------------------------

n_FFT = 2^16;       % number of FFT points
f_sam = 5*10^6;     % sampling frequency
f_in = 100*10^3;    % input frequency
v_amp_in = 4.9;     % input magnitude

% -------------------------- initialize -----------------------------------

n_sim = n_FFT;
n_bin = findBin(f_sam, f_in, n_sim);        % for coherence sampling
f_in_corr = n_bin/n_sim*f_sam;

n_DAC_Cd = n_DAC_Cd .* (1+randn([1,length(n_DAC_Cd)])./sqrt(n_DAC_Cd)*sgm_mis_cap*MISMATCH);   % add mismatch

[a_wgt_DAC, a_cap_DAC] = cap2weight(n_DAC_Cd(end:-1:1), n_DAC_Cb(end:-1:1), n_DAC_Cp(end:-1:1));    % calculate actual DAC weight
a_wgt_DAC = a_wgt_DAC(end:-1:1);

phy_KT = 1.38*10^-23 * phy_temp;  
sgm_noi_DAC = sqrt(phy_KT/(phy_cap_DAC*a_cap_DAC)) * NOISE; % kT/C noise 

v_os_cmp = randn * sgm_os_cmp * OFFSET;

% ------------------------- run simulation (single tone test) --------------

t_sim = (0:n_sim-1) / f_sam;                                % generate timeline
v_in = v_amp_in/2 * cos(t_sim * 2*pi*f_in_corr) + v_cm;     % generate input signal

d_raw = zeros([n_sim,n_SAR]);       % output raw codes

for ii_sim = 1:n_sim    % simulate each sampling point
    
    v_DAC = v_in(ii_sim) + randn * sgm_noi_DAC;   % samples input signal with kT/C noise 
    
    for ii_SAR = 1:n_SAR    % simulate each SAR cycle

        v_DAC = v_DAC - v_ref * a_wgt_DAC(ii_SAR);    % try if this bit is 1

        if(v_DAC > v_os_cmp + randn * sgm_noi_cmp)
           d_raw(ii_sim, ii_SAR) = 1;
        else
           v_DAC = v_DAC + v_ref * a_wgt_DAC(ii_SAR); % recover DAC voltage if this bit is 0
           d_raw(ii_sim, ii_SAR) = 0;
        end
        
    end
    
end

d_out_uncal = d_raw * d_wgt_nom';       % convert raw code into decimals

[a_wgt_cal,a_os_cal] = FGCalSine(d_raw,n_bin/n_sim); % calibration

d_wgt_cal = a_wgt_cal*2^12;
d_os_cal = a_os_cal*2^12;

d_out_cal = d_raw * d_wgt_cal' - d_os_cal;

% ------------------------ analyze results --------------------------------

RET = struct();

figure(1);
clf;
[ENoB,SNDR,SFDR,SNR,THD,pwr,NF] = specPlot(d_out_uncal,'Fs',f_sam,'maxCode',sum(d_wgt_nom),'harmonic',13);
title('Before calibration');

RET.SNDR_uncal = SNDR;
RET.SFDR_uncal = SFDR;
RET.THD_uncal = THD;

figure(2);
clf;
[ENoB,SNDR,SFDR,SNR,THD,pwr,NF] = specPlot(d_out_cal,'Fs',f_sam,'maxCode',sum(d_wgt_cal),'harmonic',13);
title('After calibration');

RET.SNDR = SNDR;
RET.SFDR = SFDR;
RET.THD = THD;


RET.simTime = toc();

% -------------------------------------------------------------------------
% ------------------------ inline functions  ------------------------------

% -------------------------------------------------------------------------
end