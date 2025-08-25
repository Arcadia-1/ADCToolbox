% common procedures for initializing a test

clear pr da
% close all
clc;

path_setting;   % set the data and lib path

if(~exist('hw','var'))
    clear hw
    init_hardware;  % initialize hardware if not yet
end

pr = struct();  % parameters
da = struct();  % data

BB_preset_settings;         % initialize test settings
BB_preset_reference;        % initialize supply / bias voltage / current
BB_preset_configurations;   % initialize chip configurations

% turn on vref
clear_reference_error;
if(abs(PVR_getVout(hw.pvr_Vana) - pr.Vana) > 0.01)
    update_reference;
    turnon_reference;
    pause(1);
end

% config the chip
write_reg(pr,3);

clc;
fprintf('Test Start ...\n\n');