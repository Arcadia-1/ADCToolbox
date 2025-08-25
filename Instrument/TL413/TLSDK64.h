#ifndef LASDK_SYMBOL
#define LASDK_SYMBOL

//HW Info
#define SET_TL_MODE     5

typedef char BYTE;
typedef unsigned int UINT;

enum LA_HW_MODE
{//Available Sampling Rate, Available Channel
    //LA3K hardware mode list
    LAHW_200M_128CH = 300,
    LAHW_200M_64CH,
    LAHW_200M_32CH,

    LAHW_250M_128CH,
    LAHW_250M_64CH,
    LAHW_250M_32CH,

    LAHW_500M_128CH,
    LAHW_500M_64CH,
    LAHW_500M_32CH,

    LAHW_1_0G_64CH,
    LAHW_1_0G_32CH,

    LAHW_2_0G_32CH,
    LAHW_2_4G_32CH,
    LAHW_2_0G_32CH_B,
    LAHW_2_4G_32CH_B,

    LAHW_200M_TR_128CH = 400,
    LAHW_200M_TR_64CH,
    LAHW_200M_TR_32CH,

    LAHW_250M_TR_128CH,
    LAHW_250M_TR_64CH,
    LAHW_250M_TR_32CH,

    LAHW_500M_TR_96CH,
    LAHW_500M_TR_48CH,

    LAHW_1_0G_TR_56CH,
    LAHW_1_0G_TR_24CH,

    LAHW_2_0G_TR_28CH,
    LAHW_2_4G_TR_28CH,
    LAHW_2_0G_TR_28CH_B,
    LAHW_2_4G_TR_28CH_B,

    //TL3K hardware mode list
    TLHW_200M_32CH = 1300,
    TLHW_200M_16CH,
    TLHW_200M_8CH,
    TLHW_200M_4CH,
    TLHW_250M_32CH,
    TLHW_500M_16CH,
    TLHW_1_0G_8CH,
    TLHW_2_0G_4CH,

    TLHW_SYNC_32CH = 1320,

    TLHW_200M_TR_24CH = 1400,
    TLHW_250M_TR_24CH,
    TLHW_500M_TR_12CH,
    TLHW_1_0G_TR_6CH,
    TLHW_2_0G_TR_3CH,

    TLHW_SYNC_TR_24CH = 1420,

};

//Allow Maximum 10 devices connected at same time
//But only 1 device to capture at one time
#define UPA_MAX_DEVICE  10

//Hw Configuration
#define LA_MAX_POD_NUM     4

//Trigger Setting
#define TR_CONTINUE			0
#define TR_DISCONTINUE		1
#define TR_END				2

#define TRIG_LOW			0
#define TRIG_HIGH			6
#define TRIG_DONTCARE		8

#define TL3K_MAX_TRIG_LEVEL  32
#define TL3K_MAX_CHANNEL     32

#define LA3K_MAX_TRIG_LEVEL  32
#define LA3K_MAX_CHANNEL     128


enum LA_TRIG_TYPE {
        LA_TRIG_DONT_CARE = TRIG_DONTCARE,
        LA_TRIG_LOW = TRIG_LOW,
        LA_TRIG_HIGH = TRIG_HIGH,
        LA_TRIG_RISING = 4,
        LA_TRIG_FALLING = 2,
        LA_TRIG_CHANGE = 10
};
enum LA_TRIG_NEXT {
        TR_NEXT = TR_CONTINUE,
        TR_NEXTIF = TR_DISCONTINUE,
        TR_TRIGGER = TR_END
};
//iExtClk
enum LA_EXT_CLOCK_MODE
{
    EXT_CLK_SYNC_RISE = 0,
    EXT_CLK_SYNC_FALL,
    EXT_CLK_SYNC_ANY,
};

typedef struct SDKTRIG
{
    int         iFlag;          // Trigger Flag
    int         reserved;       // Trigger Delay setting will be moved to ulaSDKSetTrigDelayByTime function
    int         reserved1;      //
    int         iPassCount;     // Trigger Pass Count
    int         iFreq;          // LA Sampling Rate
    int			iFreqHi;		// LA Sampling Rate (upper 32 bit)
    int			iExtClk;		// Enable Ext-clock
    int			iTrPos;			// Trigger Position (%), input 1~99 to set the trigger position to 1%~99%//
    int         lpiCont[16];    // Trigger Continue, the value of this parameter will be set automatically by SDK function//
    BYTE        lpbTrigData[2048];   // Trigger Value, the value of this parameter will be set automatically by SDK function//
} SDK_TRIG, *LPSDK_TRIG;


enum ENV_PARAMETER
{
    ENV_CAPTURE_QUERY_TIMER = 0x100
};

//Static link
//     #define DLLIMBOOL extern "C" __declspec(dllimport) bool __stdcall
//     #define DLLIMINT extern "C" __declspec(dllimport) int __stdcall
//     #define DLLIMUINT extern "C" __declspec(dllimport) UINT __stdcall
//     #define DLLIMI64 extern "C" __declspec(dllimport) __int64 __stdcall

    #define DLLIMBOOL bool
    #define DLLIMINT int
    #define DLLIMUINT UINT
    #define DLLIMI64 __int64

//     Initial / Shutdown / Select Device / Error Code
    DLLIMINT    ulaSDKInit();
    DLLIMBOOL   ulaSDKClose();
    DLLIMBOOL   ulaSDKSelectDevice(char* szSerialNo);
    DLLIMBOOL   ulaSDKSelectDeviceIdx(int iDeviceIndex);
    DLLIMINT    ulaSDKGetLastError();
    DLLIMBOOL   ulaSDKSetWorkDir(const char* szWorkDir, int iSize);
    
    //Query Info
    DLLIMBOOL ulaSDKGetSerialNumber(unsigned char * pBuf, int iSize);
    DLLIMINT ulaSDKGetLaID();
    
    //Set Capture Parameters
    DLLIMBOOL ulaSDKSetHwInfo(int iIndex, void* Ipv);
    DLLIMBOOL ulaSDKSetSamplesNum(UINT iSize);
    DLLIMBOOL ulaSDKThreshold(int iPod, int iMilliVolt);
    DLLIMBOOL ulaSDKSetChannelMask(BYTE *lpbChMask, int iSize);	//128bit Channel Mask
    DLLIMBOOL ulaSDKSetHwFilter(__int64 i64ChOnOff, int iFilterTime_ns);	//64bit filter setting
    
    //Set Trigger Info
    DLLIMBOOL ulaSDKSetChTrigger(LPSDK_TRIG lpSDKtr, int iLevel, int iCh, int iTrig, int iCondLogic);
    DLLIMBOOL ulaSDKClearTrigger(LPSDK_TRIG lpSDKtr);
    DLLIMBOOL ulaSDKSetExtTrigger(bool bUseExtTrig);
    DLLIMBOOL ulaSDKSetTrigDelayByTime(__int64 i64DelayTime_ns);
    
    //Capture / Stop / Ready Check
    DLLIMBOOL ulaSDKCapture( LPSDK_TRIG lpSDKtr);
    DLLIMBOOL ulaSDKStopCapture();
    DLLIMBOOL ulaSDKIsCaptureReady();
    DLLIMBOOL ulaSDKIsTriggerReady();
    
    //Query Info
    DLLIMUINT  ulaSDKGetMaxSamplesNum();
    DLLIMUINT ulaSDKGetSamplesNum();
    
    //Get Data
    DLLIMBOOL ulaSDKGetBusData(int iMSB, int iLSB, UINT* pUserData, int* lpiSize, int iStartSamplePos);
    DLLIMBOOL ulaSDKGetChData(int iCh, UINT* pUserData, int* lpiSize, int iStartSamplePos);
    //Get Transitional Storage Data
//     DLLIMBOOL ulaSDKGetTransStoreInfo(__int64 & i64TrHead, __int64 & i64TrPos);
//     DLLIMI64 ulaSDKGetTrWfm( __int64 i64Index, LPBYTE lpb );
    
    
    //Waveform Save
    DLLIMBOOL ulaSDKSaveAsLAWFile(char* szFilePathName);
    DLLIMBOOL ulaSDKSaveAsLAWFileWithTemplate(char* szFilePathName, char* szFileTemplate);
    
    //Waveform Load
    DLLIMBOOL ulaSDKLoadCaptureInfoFromFile(char *szLawFileName, LPSDK_TRIG lpSDKTrig);


//------------------------------------------------------------------------------------------------------//
//Dynamic link user may uncomment the following defines or copy them to other file to use these defines //
//------------------------------------------------------------------------------------------------------//

//Dynamic link
//     //Initial / Shutdown / Select Device / Error Code / Environment settings
//     typedef int    (__stdcall  * PULASDKINIT)();
//     typedef bool   (__stdcall  * PULASDKCLOSE)();
//     typedef bool   (__stdcall  * PULASDKSELECTDEVICE)(char* szSerialNo);
//     typedef bool   (__stdcall  * PULASDKSELECTDEVICEIDX)(int iDeviceIndex);
//     typedef int    (__stdcall  * PULASDKGETLASTERROR)();
//     typedef bool   (__stdcall  * PULASDKSETWORKDIR)(const char* szWorkDir, int iSize);
//     typedef bool   (__stdcall  * PULASDKSETENVPARAM)(int iEnv, int iParam);
// 
//     //Query Info
//     typedef bool (__stdcall  * PULASDKGETSERIALNUMBER)(char * pBuf, int iSize);
//     typedef int (__stdcall  * PULASDKGETLAID)();
// 
//     //Set Capture Parameters
//     typedef bool (__stdcall  * PULASDKSETHWINFO)(int iIndex, void* Ipv);
//     typedef bool (__stdcall  * PULASDKSETSAMPLESNUM)(UINT iSize);
//     typedef bool (__stdcall  * PULASDKTHRESHOLD)(int iPod, int iMilliVolt);
//     typedef bool (__stdcall  * PULASDKSETCHANNELMASK)(BYTE *lpbChMask, int iSize);	//128bit Channel Mask
//     typedef bool (__stdcall  * PULASDKSETHWFILTER)(__int64 i64ChOnOff, int iFilterTime_ns);	//64bit filter setting
// 
//     //Set Trigger Info
//     typedef bool (__stdcall  * PULASDKSETCHTRIGGER)(LPSDK_TRIG lpSDKtr, int iLevel, int iCh, int iTrig, int iCondLogic);
//     typedef bool (__stdcall  * PULASDKCLEARTRIGGER)(LPSDK_TRIG lpSDKtr);
//     typedef bool (__stdcall  * PULASDKSETEXTTRIGGER)(bool bUseExtTrig);
//     typedef bool (__stdcall  * PULASDKSETTRIGDELAYBYTIME)(__int64 i64DelayTime_ns);
// 
//     //Capture / Stop / Ready Check
//     typedef bool (__stdcall  * PULASDKCAPTURE)( LPSDK_TRIG lpSDKtr);
//     typedef bool (__stdcall  * PULASDKSTOPCAPTURE)();
//     typedef bool (__stdcall  * PULASDKISCAPTUREREADY)();
//     typedef bool (__stdcall  * PULASDKISTRIGGERREADY)();
// 
//     //Query Info
//     typedef UINT  (__stdcall  * PULASDKGETMAXSAMPLESNUM)();
//     typedef UINT (__stdcall  * PULASDKGETSAMPLESNUM)();
// 
//     //Get Data
//     typedef bool (__stdcall  * PULASDKGETBUSDATA)(int iMSB, int iLSB, UINT* pUserData, int* lpiSize, int iStartSamplePos);
//     typedef bool (__stdcall  * PULASDKGETCHDATA)(int iCh, UINT* pUserData, int* lpiSize, int iStartSamplePos);
//     //Get Transitional Storage Data
//     typedef bool (__stdcall  * PULASDKGETTRANSSTOREINFO)(__int64 & i64TrHead, __int64 & i64TrPos);
//     typedef __int64 (__stdcall  * PULASDKGETTRWFM)( __int64 i64Index, LPBYTE lpb );
// 
// 
//     //Waveform Save
//     typedef bool (__stdcall  * PULASDKSAVEASLAWFILE)(char* szFilePathName);
//     typedef bool (__stdcall  * PULASDKSAVEASLAWFILEWITHTEMPLATE)(char* szFilePathName, char* szFileTemplate);
// 
//     //Waveform Load
//     typedef bool (__stdcall  * PULASDKLOADCAPTUREINFOFROMFILE)(char *szLawFileName, LPSDK_TRIG lpSDKTrig);

///Dynamic link defines
    ////Initial / Shutdown / Select Device / Error Code
    //PULASDKINIT pulaSDKInit = 0;
    //PULASDKCLOSE pulaSDKClose = 0;
    //PULASDKSELECTDEVICE pulaSDKSelectDevice = 0;
    //PULASDKSELECTDEVICEIDX pulaSDKSelectDeviceIdx = 0;
    //PULASDKGETLASTERROR pulaSDKGetLastError = 0;
    //PULASDKSETWORKDIR pulaSDKSetWorkDir = 0;

    ////Query Info
    //PULASDKGETSERIALNUMBER pulaSDKGetSerialNumber = 0;
    //PULASDKGETLAID pulaSDKGetLaID = 0;

    ////Set Capture Parameters
    //PULASDKSETHWINFO pulaSDKSetHwInfo = 0;
    //PULASDKSETSAMPLESNUM pulaSDKSetSamplesNum = 0;
    //PULASDKTHRESHOLD pulaSDKThreshold = 0;
    //PULASDKSETCHANNELMASK pulaSDKSetChannelMask = 0;	//128bit Channel Mask
    //PULASDKSETHWFILTER pulaSDKSetHwFilter = 0;	//64bit filter setting

    ////Set Trigger Info
    //PULASDKSETCHTRIGGER pulaSDKSetChTrigger = 0;
    //PULASDKCLEARTRIGGER pulaSDKClearTrigger = 0;
    //PULASDKSETEXTTRIGGER pulaSDKSetExtTrigger = 0;
    //PULASDKSETTRIGDELAYBYTIME pulaSDKSetTrigDelayByTime = 0;

    ////Capture / Stop / Ready Check
    //PULASDKCAPTURE pulaSDKCapture = 0;
    //PULASDKSTOPCAPTURE pulaSDKStopCapture = 0;
    //PULASDKISCAPTUREREADY pulaSDKIsCaptureReady = 0;
    //PULASDKISTRIGGERREADY pulaSDKIsTriggerReady = 0;

    ////Query Info
    //PULASDKGETMAXSAMPLESNUM pulaSDKGetMaxSamplesNum = 0;
    //PULASDKGETSAMPLESNUM pulaSDKGetSamplesNum = 0;

    ////Get Data
    //PULASDKGETBUSDATA pulaSDKGetBusData = 0;
    //PULASDKGETCHDATA pulaSDKGetChData = 0;
    ////Get Transitional Storage Data
    //PULASDKGETTRANSSTOREINFO pulaSDKGetTransStoreInfo = 0;
    //PULASDKGETTRWFM pulaSDKGetTrWfm = 0;


    ////Waveform Save
    //PULASDKSAVEASLAWFILE pulaSDKSaveAsLAWFile = 0;
    //PULASDKSAVEASLAWFILEWITHTEMPLATE pulaSDKSaveAsLAWFileWithTemplate = 0;

    ////Waveform Load
    //PULASDKLOADCAPTUREINFOFROMFILE pulaSDKLoadCaptureInfoFromFile = 0;

///Dynamic link Sample Code
    //HMODULE ghmLASDK = NULL;
    //#ifdef _WIN64
    //		const char * szDLLName = "TLSDK64.dll";
    //#else
    //		const char * szDLLName = "TLSDK.dll";
    //#endif
    //		if( ( ghmLASDK = LoadLibrary( szDLLName ) ) == NULL )
    //{
    //    MessageBox( "Can NOT Load SDK.DLL", NULL, MB_ICONERROR );
    //}
    //else
    //{
        ////Initial / Shutdown / Select Device / Error Code
        //pulaSDKInit = (PULASDKINIT)GetProcAddress(ghmLASDK, "ulaSDKInit");
        //pulaSDKClose = (PULASDKCLOSE)GetProcAddress(ghmLASDK, "ulaSDKClose");
        //pulaSDKSelectDevice = (PULASDKSELECTDEVICE)GetProcAddress(ghmLASDK, "ulaSDKSelectDevice");
        //pulaSDKSelectDeviceIdx = (PULASDKSELECTDEVICEIDX)GetProcAddress(ghmLASDK, "ulaSDKSelectDeviceIdx");
        //pulaSDKGetLastError = (PULASDKGETLASTERROR)GetProcAddress(ghmLASDK, "ulaSDKGetLastError");
        //pulaSDKSetWorkDir = (PULASDKSETWORKDIR)GetProcAddress(ghmLASDK, "ulaSDKSetWorkDir");
        //pulaSDKSetEnvParam = (PULASDKSETENVPARAM)GetProcAddress(ghmLASDK, "ulaSDKSetEnvParam");

        ////Query Info
        //pulaSDKGetSerialNumber = (PULASDKGETSERIALNUMBER)GetProcAddress(ghmLASDK, "ulaSDKGetSerialNumber");
        //pulaSDKGetLaID = (PULASDKGETLAID)GetProcAddress(ghmLASDK, "ulaSDKGetLaID");

        ////Set Capture Parameters
        //pulaSDKSetHwInfo = (PULASDKSETHWINFO)GetProcAddress(ghmLASDK, "ulaSDKSetHwInfo");
        //pulaSDKSetSamplesNum = (PULASDKSETSAMPLESNUM)GetProcAddress(ghmLASDK, "ulaSDKSetSamplesNum");
        //pulaSDKThreshold = (PULASDKTHRESHOLD)GetProcAddress(ghmLASDK, "ulaSDKThreshold");
        //pulaSDKSetChannelMask = (PULASDKSETCHANNELMASK)GetProcAddress(ghmLASDK, "ulaSDKSetChannelMask");	//128bit Channel Mask
        //pulaSDKSetHwFilter = (PULASDKSETHWFILTER)GetProcAddress(ghmLASDK, "ulaSDKSetHwFilter");	//64bit filter setting

        ////Set Trigger Info
        //pulaSDKSetChTrigger = (PULASDKSETCHTRIGGER)GetProcAddress(ghmLASDK, "ulaSDKSetChTrigger");
        //pulaSDKClearTrigger = (PULASDKCLEARTRIGGER)GetProcAddress(ghmLASDK, "ulaSDKClearTrigger");
        //pulaSDKSetExtTrigger = (PULASDKSETEXTTRIGGER)GetProcAddress(ghmLASDK, "ulaSDKSetExtTrigger");
        //pulaSDKSetTrigDelayByTime = (PULASDKSETTRIGDELAYBYTIME)GetProcAddress(ghmLASDK, "ulaSDKSetTrigDelayByTime");

        ////Capture / Stop / Ready Check
        //pulaSDKCapture = (PULASDKCAPTURE)GetProcAddress(ghmLASDK, "ulaSDKCapture");
        //pulaSDKStopCapture = (PULASDKSTOPCAPTURE)GetProcAddress(ghmLASDK, "ulaSDKStopCapture");
        //pulaSDKIsCaptureReady = (PULASDKISCAPTUREREADY)GetProcAddress(ghmLASDK, "ulaSDKIsCaptureReady");
        //pulaSDKIsTriggerReady = (PULASDKISTRIGGERREADY)GetProcAddress(ghmLASDK, "ulaSDKIsTriggerReady");

        ////Query Info
        //pulaSDKGetMaxSamplesNum = (PULASDKGETMAXSAMPLESNUM)GetProcAddress(ghmLASDK, "ulaSDKGetMaxSamplesNum");
        //pulaSDKGetSamplesNum = (PULASDKGETSAMPLESNUM)GetProcAddress(ghmLASDK, "ulaSDKGetSamplesNum");

        ////Get Data
        //pulaSDKGetBusData = (PULASDKGETBUSDATA)GetProcAddress(ghmLASDK, "ulaSDKGetBusData");
        //pulaSDKGetChData = (PULASDKGETCHDATA)GetProcAddress(ghmLASDK, "ulaSDKGetChData");
        ////Get Transitional Storage Data
        //pulaSDKGetTransStoreInfo = (PULASDKGETTRANSSTOREINFO)GetProcAddress(ghmLASDK, "ulaSDKGetTransStoreInfo");
        //pulaSDKGetTrWfm = (PULASDKGETTRWFM)GetProcAddress(ghmLASDK, "ulaSDKGetTrWfm");


        ////Waveform Save
        //pulaSDKSaveAsLAWFile = (PULASDKSAVEASLAWFILE)GetProcAddress(ghmLASDK, "ulaSDKSaveAsLAWFile");
        //pulaSDKSaveAsLAWFileWithTemplate = (PULASDKSAVEASLAWFILEWITHTEMPLATE)GetProcAddress(ghmLASDK, "ulaSDKSaveAsLAWFileWithTemplate");

        ////Waveform Load
        //pulaSDKLoadCaptureInfoFromFile = (PULASDKLOADCAPTUREINFOFROMFILE)GetProcAddress(ghmLASDK, "ulaSDKLoadCaptureInfoFromFile");
    //}

///LabVIEW import DLL
        //Initial / Shutdown / Select Device / Error Code
        //int    ulaSDKInit();
        //bool   ulaSDKClose();
        //bool   ulaSDKSelectDevice(char* szSerialNo);
        //bool   ulaSDKSelectDeviceIdx(int iDeviceIndex);
        //int    ulaSDKGetLastError();
        //bool   ulaSDKSetWorkDir(const char* szWorkDir, int iSize);
        //
        ////Query Info
        //bool ulaSDKGetSerialNumber(char * pBuf, int iSize);
        //int ulaSDKGetLaID();
        //
        ////Set Capture Parameters
        //bool ulaSDKSetHwInfo(int iIndex, void* Ipv);
        //bool ulaSDKSetSamplesNum(int iSize);
        //bool ulaSDKThreshold(int iPod, int iMilliVolt);
        //bool ulaSDKSetChannelMask(BYTE *lpbChMask, int iSize);	//128bit Channel Mask
        //bool ulaSDKSetHwFilter(__int64 i64ChOnOff, int iFilterTime_ns);	//64bit filter setting
        //
        ////Set Trigger Info
        //bool ulaSDKSetChTrigger(LPSDK_TRIG lpSDKtr, int iLevel, int iCh, int iTrig, int iCondLogic);
        //bool ulaSDKClearTrigger(LPSDK_TRIG lpSDKtr);
        //bool ulaSDKSetExtTrigger(bool bUseExtTrig);
        //bool ulaSDKSetTrigDelayByTime(__int64 i64DelayTime_ns);
        //
        ////Capture / Stop / Ready Check
        //bool ulaSDKCapture( LPSDK_TRIG lpSDKtr);
        //bool ulaSDKStopCapture();
        //bool ulaSDKIsCaptureReady();
        //bool ulaSDKIsTriggerReady();
        //
        ////Query Info
        //int ulaSDKGetMaxSamplesNum();
        //bool ulaSDKGetSamplesNum();
        //
        ////Get Data
        //bool ulaSDKGetBusData(int iMSB, int iLSB, UINT* pUserData, int* lpiSize, int iStartSamplePos);
        //bool ulaSDKGetChData(int iCh, UINT* pUserData, int* lpiSize, int iStartSamplePos);
        ////Get Transitional Storage Data
        //bool ulaSDKGetTransStoreInfo(__int64 & i64TrHead, __int64 & i64TrPos);
        //__int64 ulaSDKGetTrWfm( __int64 i64Index, LPBYTE lpb );
        //
        //
        ////Waveform Save
        //bool ulaSDKSaveAsLAWFile(char* szFilePathName);
        //bool ulaSDKSaveAsLAWFileWithTemplate(char* szFilePathName, char* szFileTemplate);
        //
        ////Waveform Load
        //bool ulaSDKLoadCaptureInfoFromFile(char *szLawFileName, LPSDK_TRIG lpSDKTrig);
#endif // LASDK_SYMBOL

