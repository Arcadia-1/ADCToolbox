function keysight = Keysight_PSG_init(ip)

    keysight = tcpip(ip,5025);
    keysight.OutputBufferSize = 100000;
    keysight.ByteOrder = 'littleEndian';
    keysight.Timeout = 10.0;

end