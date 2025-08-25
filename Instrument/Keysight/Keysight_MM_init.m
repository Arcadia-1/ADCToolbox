function keysight = Keysight_MM_init(sn)
% This function sets up communication with a Keysight multimeter using its serial number,
% resets the device to its default state, and checks its status by running a self-test.
%
% Author: Zhishuai Zhang
% Date: 2024.8.13
%
% Parameters:
%   sn - Serial number of the Keysight device.
%
% Returns:
%   keysight - The VISA device object for the Keysight instrument.
%
% Example of use:
%   Keysight_MM_5 = 'USB0::0x2A8D::0x0201::MY57701960::INSTR';
%   keysight = Keysight_MM_init(Keysight_MM_5);
%
% Note: Ensure that the 'visadev', 'writeline', and 'writeread' functions are defined
%       and available. Instrument Control Toolbox is required.

keysight = visadev(sn);

writeline(keysight, '*RST')
writeread(keysight, 'TEST:ALL?') % Runs an instrument self-test and returns a pass/fail indication.


end