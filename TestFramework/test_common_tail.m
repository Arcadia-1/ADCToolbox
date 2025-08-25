TL413_stop();   % force stop logic analyzer capture 

APX_turn(hw.apx,0);     % turn off source
RIGOL_SS_turn(hw.rigol_ss, 0, 0);

wait(0.1);

da.regVerify = verify_reg(pr,3);    % double check if the chip is config correctly

% turnoff_reference;    % turn off power (optional)

% save all parameters and data
save(sprintf('%s\\%s_chip%d_%s_%s.mat', save_path, project_name, pr.chip_id, pr.test_type, datetime('now','format','yyyyMMddHHmmss')),'pr','da','-v7.3');
fprintf('Data Recorded.\n\n');

fprintf('Test Finished.\n\n');
