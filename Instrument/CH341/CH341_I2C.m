function [dat_read] = CH341_I2C(addr, dat_write, n_read, device_id)

% addr: I2C address
% dat_write: the data write by I2C, byte array
% n_read: the number of bytes read by I2C

    if(nargin < 4)
        device_id = 0;
    end
    if(nargin < 3)
        n_read = 0;
    end
    
    if(addr > 255 || addr < 0)
        error('CH341: Incorrect I2C address');
    end
    if(max(dat_write) > 255 || min(dat_write) < 0)
        error('CH341: Incorrect I2C data sent');
    end
    if(n_read < 0)
        error('CH341: Incorrect I2C number of data read');
    end
    
    CH341_check(device_id);
    
    calllib('CH341DLLA64','CH341OpenDevice',device_id);

    iWriteBuffer = libpointer('uint8Ptr', [addr, dat_write]);
    iReadBuffer = libpointer('uint8Ptr', zeros([1,n_read]));
    calllib('CH341DLLA64','CH341StreamI2C',device_id,1+length(dat_write),iWriteBuffer,n_read,iReadBuffer);
    if(n_read == 0)
        dat_read = [];
    else
        dat_read = iReadBuffer.value;
    end
    clear iWriteBuffer
    clear iReadBuffer
    calllib('CH341DLLA64','CH341CloseDevice',device_id);
end