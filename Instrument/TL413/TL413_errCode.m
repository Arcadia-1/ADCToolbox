function s = TL413_errCode(iErrorCode)

switch iErrorCode

case 0
	s = 'LA_SDK_ERR_OK';
%Success, No Error!!%

case 1 
	s = 'LA_SDK_ERR_LA_NOT_INITIAL';
%Need to call ulaSDKInit('; to initiate LA before calling other funciton%


case 16 
	s = 'LA_SDK_ERR_TRIGINFO';
%Sample Rate Can't be 0%


case 17 
	s = 'LA_SDK_ERR_TOO_MANY_TRIG_CONDITION';
%Too many Trigger condition, max level is 16					%
%Decrease to 8 if rising edge / falling edge condition used	%
%Decrease to 4 if Change edge condition used					%


case 18
	s = 'LA_SDK_ERR_CH_NUM_OUT_OF_RANGE';
%Input Ch number is out of range%

case 19 
	s = 'LA_SDK_ERR_TRIGSET_CANNOT_BUILD';
%Trig set build error, builded trig set over max level %
%Clear the trig set and reduce the trig condition		%

case 20
	s = 'LA_SDK_ERR_INPUT_BAD_POINTER';
%Input pointer variable is a bad pointer%

case 21 
	s = 'LA_SDK_ERR_INPUT_INVALID_CONDITION';
%Input Trig condition is invalid, reference the following table%
%LA_TRIG_DONT_CARE, LA_TRIG_LOW, LA_TRIG_HIGH, LA_TRIG_RISING, LA_TRIG_FALLING, LA_TRIG_CHANGE%

case 22 
	s = 'LA_SDK_ERR_DEVICE_NOT_SUPPORT';
%Current device is not support the Rising / Falling /Either Edge condition%

case 23 
	s = 'LA_SDK_ERR_GET_LA_DATA_ERR';
%Request data from LA failed, check the USB connection%

case 24
	s = 'LA_SDK_ERR_SAMPLE_NUM_OUT_OF_RANGE';
%Specified Sample Number is out of range%

case 25 
	s = 'LA_SDK_ERR_HW_MODE_OUT_OF_RANGE';
%Specified HW Mode is out of range%

case 32 
	s = 'LA_SDK_ERR_INPUT_VAR_IS_NULL';
%Input Variable(Pointer'; is NULL%

case 33	 
	s = 'LA_SDK_ERR_VAR_NOT_READY';
%Need to call ulaSDKCaputre('; to get datas before ulaSDKSaveAsLawFile(';%

case 34
	s = 'LA_SDK_ERR_FILE_OPEN_FAIL';
%Save File Create Error, target dir is Protected or insufficient disk space%

case 35
	s = 'LA_SDK_ERR_STATUS_GET_ERROR';
%Cannot get La Current HW mode %

case 36 
	s = 'LA_SDK_ERR_FILE_FORMAT_ERROR';
%Target file format unknown%

case 37
	s = 'LA_SDK_ERR_FILE_READ_FAIL';
%LAW file read error, file does not exist or is protected%

case 38 
	s = 'LA_SDK_ERR_OLD_VERSION';
%Specified a higher version LAW file, please contact Acute for new SDK version%

case 39
	s = 'LA_SDK_ERR_BUFFER_TOO_SMALL';
%Buffer size too small(<= 0';%

case 40 
	s = 'LA_SDK_ERR_FW_TOO_OLD';
%Firmware too old, please contact Acute%

case 41 
	s = 'LA_SDK_ERR_DELAYTRIG_OUT_OF_RANGE';
    %Delay Trigger is limited from 0 to 0xffffff clocks, please check the
    %input value%
	

case 48 
	s = 'LA_SDK_ERR_CAPTURE_IN_PROGRESS';
%The LA is still in capture operation%


case 49
	s = 'LA_SDK_ERR_CAPTURE_START_FAILED';
%Unable to initiate the LA capture operation%

case 50
	s = 'LA_SDK_ERR_PROCDURE_START_FAILED';
%Unable to start a new procedure%
        
case 2456 
	s = 'LA_SDK_ERR_TRIG_FORMAT_CONV_ERROR';
%Internal Trigger format convert error, contact Acute%

case 2457 
	s = 'LA_SDK_ERR_MEMORY_ACCESS_ERROR';
%Internal memory copy error, contact Acute

otherwise
    s = 'UNKNOWN_ERROR(Please refer to TLSDK64_err.h)';

end