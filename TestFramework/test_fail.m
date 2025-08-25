% this is to terminate a AIO test when fatal errors occur
% all data will be stil saved, after that the test is stop and shutdown

da_AIO.success = 0;

measure_reference;
record_reference_error;

da_AIO.time_end = datetime;

save(sprintf('%s\\%s_chip%d_AllInOne_Failed_%s.mat', save_path, project_name, pr_AIO.chip_id, datetime('now','format','yyyyMMddHHmmss')),'pr_AIO','da_AIO','da','pr','-v7.3');
fprintf('\n\nData Recorded.\n\n');

clear pr_AIO

fprintf('Shuting Down and exit...\n\n');

shutdown;

error('All in One Test Failed with Error: %s',da_AIO.err);