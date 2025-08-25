function apx = APX_2722_init()
% not yet complete
%     apiRoot = 'C:\Program Files (x86)\Audio Precision\APx500 4.4\API\';
    NET.addAssembly([apiRoot 'AudioPrecision.API2.dll']);
    apx = AudioPrecision.API.APx500_Application;
end
