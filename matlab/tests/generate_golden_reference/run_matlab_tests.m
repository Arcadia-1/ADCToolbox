%% Run tests to generate golden reference data
% Run golden tests - only processes files from golden_data_list.txt
% Outputs are saved to test_output/

% Run golden tests from this folder
golden_sineFit

% Run test_alias from unit tests folder (doesn't use datasets)
cd ../unit
test_alias
cd ../generate_golden_reference
